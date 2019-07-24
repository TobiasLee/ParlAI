#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .utils_v0 import Beam
from .dict_v0 import DictionaryAgent
import numpy as np
import os


def pad(tensor, length, dim=0):
    if tensor.size(dim) < length:
        return torch.cat(
            [tensor, tensor.new(*tensor.size()[:dim],
                                length - tensor.size(dim),
                                *tensor.size()[dim + 1:]).zero_()],
            dim=dim)
    else:
        return tensor


class Seq2seq(nn.Module):
    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, opt, num_features,
                 padding_idx=0, start_idx=1, end_idx=2, longest_label=1,):
        super().__init__()
        self.opt = opt

        self.rank = opt['rank_candidates']
        self.attn_type = opt['attention']

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        rnn_class = Seq2seq.RNN_OPTS[opt['rnn_class']]
        self.decoder = Decoder(
            num_features, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=opt['embeddingsize'], hidden_size=opt['hiddensize'],
            num_layers=opt['numlayers'], dropout=opt['dropout'],
            share_output=opt['lookuptable'] in ['dec_out', 'all'],
            attn_type=opt['attention'], attn_length=opt['attention_length'],
            attn_time=opt.get('attention_time'),
            bidir_input=opt['bidirectional'],
            numsoftmax=opt.get('numsoftmax', 1),
            softmax_layer_bias=opt.get('softmax_layer_bias', False))

        shared_lt = (self.decoder.lt
                     if opt['lookuptable'] in ['enc_dec', 'all'] else None)
        shared_rnn = self.decoder.rnn if opt['decoder'] == 'shared' else None
        self.encoder = Encoder(
            opt=opt, num_features=num_features, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=opt['embeddingsize'], hidden_size=opt['hiddensize'],
            num_layers=opt['numlayers'], dropout=opt['dropout'],
            bidirectional=opt['bidirectional'],
            shared_lt=shared_lt, shared_rnn=shared_rnn)

        if self.rank:
            self.ranker = Ranker(
                self.decoder,
                padding_idx=self.NULL_IDX,
                attn_type=opt['attention'])

        self.beam_log_freq = opt.get('beam_log_freq', 0.0)
        if self.beam_log_freq > 0.0:
            self.dict = DictionaryAgent(opt)
            self.beam_dump_filecnt = 0
            self.beam_dump_path = opt['model_file'] + '.beam_dump'
            if not os.path.exists(self.beam_dump_path):
                os.makedirs(self.beam_dump_path)

    def unbeamize_hidden(self, hidden, beam_size, batch_size):
        """
        Creates a view of the hidden where batch axis is collapsed with beam axis,
        we need to do this for batched beam search, i.e. we emulate bigger mini-batch
        :param hidden: hidden state of the decoder
        :param beam_size: beam size, i.e. num of hypothesis
        :param batch_size: number of samples in the mini batch
        :return: view of the hidden
        """
        if isinstance(hidden, tuple):
            num_layers = hidden[0].size(0)
            hidden_size = hidden[0].size(-1)
            return (hidden[0].view(num_layers, batch_size * beam_size, hidden_size),
                hidden[1].view(num_layers, batch_size * beam_size, hidden_size))
        else:  # GRU
            num_layers = hidden.size(0)
            hidden_size = hidden.size(-1)
            return hidden.view(num_layers, batch_size * beam_size, hidden_size)

    def unbeamize_enc_out(self, enc_out, beam_size, batch_size):
        hidden_size = enc_out.size(-1)
        return enc_out.view(batch_size * beam_size, -1, hidden_size)

    def forward(self, xs, ys=None, qs=None, q_invert_index=None, cands=None, valid_cands=None, prev_enc=None,
                rank_during_training=False, beam_size=1, topk=1, qni=None, qni_invert_index=None, start_rl=False):
        """Get output predictions from the model.

        Arguments:
        xs -- input to the encoder
        ys -- expected output from the decoder
        cands -- set of candidates to rank, if applicable
        valid_cands -- indices to match candidates with their appropriate xs
        prev_enc -- if you know you'll pass in the same xs multiple times and
            the model is in eval mode, you can pass in the encoder output from
            the last forward pass to skip recalcuating the same encoder output
        rank_during_training -- (default False) if set, ranks any available
            cands during training as well
        """
        input_xs = xs
        nbest_beam_preds, nbest_beam_scores = None, None
        bsz = len(xs)
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        if prev_enc is not None:
            enc_out, hidden, attn_mask, log_p, actions, percentage = prev_enc
        else:
            enc_out, hidden, attn_mask, log_p, actions, percentage = \
                self.encoder(xs=xs, qs=qs, q_invert_index=q_invert_index, qni=qni, qni_invert_index=qni_invert_index,
                             start_rl=start_rl)
            if not self.opt['rl']:
                attn_mask = xs.ne(0).float()
            attn_mask = attn_mask if self.attn_type != 'none' else None
        encoder_states = (enc_out, hidden, attn_mask)
        start = self.START.detach()
        starts = start.expand(bsz, 1)

        predictions = []
        scores = []
        cand_preds, cand_scores = None, None
        if self.rank and cands is not None:
            decode_params = (start, hidden, enc_out, attn_mask)
            if self.training:
                if rank_during_training:
                    cand_preds, cand_scores = self.ranker.forward(cands, valid_cands, decode_params=decode_params)
            else:
                cand_preds, cand_scores = self.ranker.forward(cands, valid_cands, decode_params=decode_params)

        if ys is not None:
            y_in = ys.narrow(1, 0, ys.size(1) - 1)
            xs = torch.cat([starts, y_in], 1)
            if self.attn_type == 'none':
                preds, score, hidden = self.decoder(xs, hidden, enc_out, attn_mask)
                predictions.append(preds)
                scores.append(score)
            else:
                for i in range(ys.size(1)):
                    xi = xs.select(1, i)
                    preds, score, hidden = self.decoder(xi, hidden, enc_out, attn_mask)
                    predictions.append(preds)
                    scores.append(score)
        else:
            # here we do search: supported search types: greedy, beam search
            if beam_size == 1:
                done = [False for _ in range(bsz)]
                total_done = 0
                xs = starts

                for _ in range(self.longest_label):
                    # generate at most longest_label tokens
                    preds, score, hidden = self.decoder(xs, hidden, enc_out, attn_mask, topk)
                    scores.append(score)
                    xs = preds
                    predictions.append(preds)

                    # check if we've produced the end token
                    for b in range(bsz):
                        if not done[b]:
                            # only add more tokens for examples that aren't done
                            if preds.data[b][0] == self.END_IDX:
                                # if we produced END, we're done
                                done[b] = True
                                total_done += 1
                    if total_done == bsz:
                        # no need to generate any more
                        break

            elif beam_size > 1:
                enc_out, hidden = encoder_states[0], encoder_states[1]  # take it from encoder
                enc_out = enc_out.unsqueeze(1).repeat(1, beam_size, 1, 1)
                # create batch size num of beams
                data_device = enc_out.device
                beams = [Beam(beam_size, 3, 0, 1, 2, min_n_best=beam_size / 2, cuda=data_device) for _ in range(bsz)]
                # init the input with start token
                xs = starts
                # repeat tensors to support batched beam
                xs = xs.repeat(1, beam_size)
                attn_mask = input_xs.ne(0).float()
                attn_mask = attn_mask.unsqueeze(1).repeat(1, beam_size, 1)
                repeated_hidden = []

                if isinstance(hidden, tuple):
                    for i in range(len(hidden)):
                        repeated_hidden.append(hidden[i].unsqueeze(2).repeat(1, 1, beam_size, 1))
                    hidden = self.unbeamize_hidden(tuple(repeated_hidden), beam_size, bsz)
                else:  # GRU
                    repeated_hidden = hidden.unsqueeze(2).repeat(1, 1, beam_size, 1)
                    hidden = self.unbeamize_hidden(repeated_hidden, beam_size, bsz)
                enc_out = self.unbeamize_enc_out(enc_out, beam_size, bsz)
                xs = xs.view(bsz * beam_size, -1)
                for step in range(self.longest_label):
                    if all((b.done() for b in beams)):
                        break
                    out = self.decoder(xs, hidden, enc_out)
                    scores = out[1]
                    scores = scores.view(bsz, beam_size, -1)  # -1 is a vocab size
                    for i, b in enumerate(beams):
                        b.advance(F.log_softmax(scores[i, :], dim=-1))
                    xs = torch.cat([b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
                    permute_hidden_idx = torch.cat(
                        [beam_size * i + b.get_backtrack_from_current_step() for i, b in enumerate(beams)])
                    new_hidden = out[2]
                    if isinstance(hidden, tuple):
                        for i in range(len(hidden)):
                            hidden[i].data.copy_(new_hidden[i].data.index_select(dim=1, index=permute_hidden_idx))
                    else:  # GRU
                        hidden.data.copy_(new_hidden.data.index_select(dim=1, index=permute_hidden_idx))

                for b in beams:
                    b.check_finished()
                beam_pred = [b.get_pretty_hypothesis(b.get_top_hyp()[0])[1:] for b in beams]
                # these beam scores are rescored with length penalty!
                beam_scores = torch.stack([b.get_top_hyp()[1] for b in beams])
                pad_length = max([t.size(0) for t in beam_pred])
                beam_pred = torch.stack([pad(t, length=pad_length, dim=0) for t in beam_pred], dim=0)

                #  prepare n best list for each beam
                n_best_beam_tails = [b.get_rescored_finished(n_best=len(b.finished)) for b in beams]
                nbest_beam_scores = []
                nbest_beam_preds = []
                for i, beamtails in enumerate(n_best_beam_tails):
                    perbeam_preds = []
                    perbeam_scores = []
                    for tail in beamtails:
                        perbeam_preds.append(beams[i].get_pretty_hypothesis(beams[i].get_hyp_from_finished(tail)))
                        perbeam_scores.append(tail.score)
                    nbest_beam_scores.append(perbeam_scores)
                    nbest_beam_preds.append(perbeam_preds)

                if self.beam_log_freq > 0.0:
                    num_dump = round(bsz * self.beam_log_freq)
                    for i in range(num_dump):
                        dot_graph = beams[i].get_beam_dot(dictionary=self.dict)
                        dot_graph.write_png(os.path.join(self.beam_dump_path, "{}.png".format(self.beam_dump_filecnt)))
                        self.beam_dump_filecnt += 1

                predictions = beam_pred
                scores = beam_scores

        if isinstance(predictions, list):
            predictions = torch.cat(predictions, 1)
        if isinstance(scores, list):
            scores = torch.cat(scores, 1)
        return predictions, scores, cand_preds, cand_scores, encoder_states, nbest_beam_preds, nbest_beam_scores, log_p, actions, percentage


class Encoder(nn.Module):
    def __init__(self, opt, num_features, padding_idx=0, rnn_class='lstm',
                 emb_size=128, hidden_size=128, num_layers=2, dropout=0.1,
                 bidirectional=False, shared_lt=None, shared_rnn=None,
                 sparse=False):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layers = num_layers
        self.dirs = 2 if bidirectional else 1
        self.hsz = hidden_size
        self.select_ref = opt['select_ref']
        self.rl = opt['rl']  # use rl to select context default false

        self.separate_query = opt['separate_query']
        self.is_normal = self.separate_query not in ['qes', 'q_hidden', 'q_output', 'xes', 'x_output','x_hidden'] \
                         and not self.rl
        self.fuse_hidden_size = opt['fuse_hidden_size']


        if shared_lt is None:
            self.lt = nn.Embedding(num_features, emb_size,
                                   padding_idx=padding_idx,
                                   sparse=sparse)
        else:
            self.lt = shared_lt
        # self.lt_query = nn.Embedding(num_features, emb_size, padding_idx=padding_idx, sparse=sparse)

        if shared_rnn is None:
            self.rnn = rnn_class(emb_size, hidden_size, num_layers,
                                 dropout=dropout, batch_first=True,
                                 bidirectional=bidirectional)
        elif bidirectional:
            raise RuntimeError('Cannot share decoder with bidir encoder.')
        else:
            self.rnn = shared_rnn

        if self.select_ref in ['q_hidden','q_output']:
            if opt['share_q_rnn']:
                self.q_rnn = self.rnn
                self.fuse_hidden_size = hidden_size
            else:
                self.q_rnn = rnn_class(emb_size, self.fuse_hidden_size, num_layers,
                                       dropout=dropout, batch_first=True,
                                       bidirectional=bidirectional)

        if self.select_ref in ['x_output', 'x_hidden']:
            if opt['share_q_rnn']:
                self.x_rnn = self.rnn
                self.fuse_hidden_size = hidden_size
            else:
                self.x_rnn = rnn_class(emb_size, self.fuse_hidden_size, num_layers,
                                       dropout=dropout, batch_first=True,
                                       bidirectional=bidirectional)
        if self.select_ref in ['q_hidden', 'q_output', 'x_output', 'x_hidden']:
            if opt['fc3_relu']:
                self.fc3 = nn.Sequential(nn.Linear(self.fuse_hidden_size, emb_size),
                                         nn.ReLU())
            else:
                self.fc3 = nn.Linear(self.fuse_hidden_size, emb_size)
        if opt['fc1_relu']:
            self.fc1 = nn.Sequential(nn.Linear(2 * emb_size, emb_size),
                                     nn.ReLU())
        else:
            self.fc1 = nn.Linear(2 * emb_size, emb_size)
        self.fc2 = nn.Linear(emb_size, 1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def select_context_word_real(self, qni_es, qni_mask):
        """
        select context by rl
        :param qni_es [bsz,m,d]
        :param qni_mask [bsz,m]
        :return:
        """
        # probs = F.sigmoid(self.dropout(F.relu(self.fc2(self.layer_norm(qni_es)))))
        # print('qni es size()',qni_es.size()) # torch.Size([64, 105, 256])
        # print('qni mask size()',qni_mask.size()) # torch.Size([64, 105])
        probs = F.sigmoid(self.fc2(qni_es)).squeeze(2)  # [64, 105]
        actions = torch.gt(probs, torch.rand_like(probs))
        actions = actions.clone().detach()  # bsz,m
        # print('actions size()', actions.size())
        actions = actions.float() * qni_mask
        log_p = -F.binary_cross_entropy(input=probs, target=actions) * qni_mask
        percentage = torch.div(torch.sum(input=actions, dim=-1), torch.sum(input=qni_mask, dim=-1))
        return log_p, actions, percentage

    def select_context_word_fake(self, qni_mask):
        """
        select all context
        :param qni_mask [bsz,m]
        :return:
        """
        log_p = torch.ones_like(qni_mask).float()
        actions = qni_mask
        percentage = torch.ones(qni_mask.size()[0])
        return log_p, actions, percentage

    def select_context_word(self, qni_es, qni_mask, start_rl):
        if start_rl:
            log_p, actions, percentage = self.select_context_word_real(qni_es=qni_es, qni_mask=qni_mask,)
        else:
            log_p, actions, percentage = self.select_context_word_fake(qni_mask=qni_mask, )
        return log_p, actions, percentage

    def forward(self, xs, qs=None, q_invert_index=None, qni=None, qni_invert_index=None, start_rl=False):
        """
        :param xs: whole context, if sq only contain history,else contain history and text
        :param qs: query for mine useful info in context
        :param q_invert_index: a LongTensor that map  sorted query encode result back to context_sorted order
        :param qni: context but query not included
        :param qni_invert_index: a LongTensor that map  sorted qni encode result back to context_sorted order
        :param start_rl: whether start to train rl and cross entropy simultaneously
        :return:
        """
        bsz = len(xs)
        attn_mask = xs.ne(0).float()
        # embed input tokens
        # x_lens_origin = [x for x in torch.sum((xs > 0).int(), dim=1).data]
        if self.is_normal or(qs is None and q_invert_index is None and qni is None and qni_invert_index is None):
            xes = self.dropout(self.lt(xs))
            try:
                x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
                xes = pack_padded_sequence(xes, x_lens, batch_first=True)
                packed = True
            except ValueError:
                # packing failed, don't pack then
                packed = False

            encoder_output, hidden = self.rnn(xes)
            if packed:
                encoder_output, _ = pad_packed_sequence(encoder_output,
                                                        batch_first=True)
            if self.dirs > 1:
                # take elementwise max between forward and backward hidden states
                # NOTE: currently using max, but maybe should use Linear
                if isinstance(self.rnn, nn.LSTM):
                    hidden = (hidden[0].view(-1, self.dirs, bsz, self.hsz).max(1)[0],
                              hidden[1].view(-1, self.dirs, bsz, self.hsz).max(1)[0])
                else:
                    hidden = hidden.view(-1, self.dirs, bsz, self.hsz).max(1)[0]
            log_p = None
            actions = None
            percentage = None
        else:
            # if self.separate_query: default false because  i am reluctant to concat query and history before decoding
            qni = torch.index_select(qni, 0, qni_invert_index)  # bsz , m x order
            qni_es = self.dropout(self.lt(qni))  # x order
            # q_lens e.g.[10,6,8,15] sorted q_lens [15, 10, 8, 6]
            # q_index_sorted list of index e.g. [3,0,2,1] the 3's position of the origin qs is longest
            # q_invert_index e.g. [1,3,2,0]
            # index_select back to origin unsorted order
            qni_mask = qni.ne(0).float()  # bsz ,m
            qni_mask_unsqueezed = torch.unsqueeze(qni_mask, 2)  # bsz , m, 1
            if self.select_ref in ['qes', 'q_hidden', 'q_output']:
                if self.select_ref == 'qes':
                    qs = torch.index_select(qs, 0, q_invert_index)  # bsz , n x order
                    qes = self.dropout(self.lt(qs))  # [bsz,query_len,emb_size] # x order
                    qs_mask = qs.ne(0).float()  # origin order
                    qs_mask_unsqueezed = torch.unsqueeze(qs_mask, 1)  # bsz , 1, n
                    mask = torch.bmm(qni_mask_unsqueezed, qs_mask_unsqueezed)
                    attention_context_query = torch.bmm(qni_es, qes.transpose(1, 2))  # [bsz, m, n]
                    attention_context_query -= (1 - mask) * 1e20
                    # attention_context_query = torch.add(input=attention_context_query, value=-1 * 1e20,
                    #                                     other=torch.add(torch.neg(mask), 1))
                    attention_context_query_score = self.softmax(attention_context_query)
                    xq = torch.bmm(attention_context_query_score, qes)  # [bsz,m,e]
                else:  # use rnn to encode qs
                    q_lens = [q for q in torch.sum((qs > 0).int(), dim=1).data]  # sorte order
                    qes = self.dropout(self.lt(qs))  # [bsz,query_len,emb_size] # sorted order
                    qes_pack_padded = qes.clone()
                    try:
                        qes_pack_padded = pack_padded_sequence(qes_pack_padded, q_lens, batch_first=True)
                        packed = True
                    except ValueError:
                        # packing failed, don't pack then
                        packed = False
                    q_encoder_output, q_hidden = self.q_rnn(qes_pack_padded)  # bsz*sq_len, hidden_size
                    # back to x order
                    qs = torch.index_select(qs, 0, q_invert_index)  # bsz , n origin order
                    qs_mask = qs.ne(0).float()  # origin order
                    qs_mask_unsqueezed = torch.unsqueeze(qs_mask, 1)  # bsz , 1, n
                    mask = torch.bmm(qni_mask_unsqueezed, qs_mask_unsqueezed)
                    # todo i don't know if lt(qs) could save time and memory  and time efficiency of index select
                    qes = torch.index_select(qes, 0, q_invert_index)  # bsz , n ,d
                    if self.select_ref == 'q_output':  # use rnn output of qs
                        if packed:
                            q_encoder_output, _ = pad_packed_sequence(q_encoder_output, batch_first=True)
                        q_encoder_output = torch.index_select(q_encoder_output, 0, q_invert_index)
                        q_encoder_output = self.fc3(q_encoder_output)
                        attention_context_query = torch.bmm(qni_es, q_encoder_output.transpose(1, 2))  # [bsz, m, n]
                        attention_context_query -= (1 - mask) * 1e20
                        # attention_context_query = torch.add(input=attention_context_query, value=-1 * 1e20,
                        #                                     other=torch.add(torch.neg(mask), 1))
                        attention_context_query_score = self.softmax(attention_context_query)
                        xq = torch.bmm(attention_context_query_score, q_encoder_output)  # [bsz,m,h]
                    else:  # use qs rnn last hidden
                        if self.dirs > 1:
                            # take elementwise max between forward and backward hidden states
                            # NOTE: currently using max, but maybe should use Linear
                            if isinstance(self.q_rnn, nn.LSTM):
                                q_hidden = (q_hidden[0].view(-1, self.dirs, bsz, self.hsz).max(1)[0],
                                            q_hidden[1].view(-1, self.dirs, bsz, self.hsz).max(1)[0])
                            else:
                                q_hidden = q_hidden.view(-1, self.dirs, bsz, self.hsz).max(1)[0]
                        if type(q_hidden) == tuple:
                            # for lstms use the "hidden" state not the cell state
                            q_hidden = q_hidden[0]
                        q_hidden = q_hidden[-1]
                        q_hidden = torch.index_select(q_hidden, 0, q_invert_index)  # back to context order
                        q_hidden = self.fc3(q_hidden)
                        xq = q_hidden.unsqueeze(1).expand_as(qni_es)  # assert emb_size= hsz

            else:
                qs = torch.index_select(qs, 0, q_invert_index)
                qs_mask = qs.ne(0).float()  # origin order
                if not self.rl:
                    qes = self.dropout(self.lt(qs))  # [bsz,query_len,emb_size] # x order
                xes_a = self.dropout(self.lt(xs))  # additional
                xs_mask = xs.ne(0).float()  # origin order
                xs_mask_unsqueezed = torch.unsqueeze(xs_mask, 1)  # bsz , 1, n
                mask = torch.bmm(qni_mask_unsqueezed, xs_mask_unsqueezed)
                if self.select_ref == 'xes':
                    attention_context_query = torch.bmm(qni_es, xes_a.transpose(1, 2))  # [bsz, m, n]
                    attention_context_query -= (1 - mask) * 1e20
                    # attention_context_query = torch.add(input=attention_context_query, value=-1 * 1e20,
                    #                                     other=torch.add(torch.neg(mask), 1))
                    attention_context_query_score = self.softmax(attention_context_query)
                    xq = torch.bmm(attention_context_query_score, xes_a)  # [bsz,m,e]
                else:
                    try:
                        x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
                        xes_a = pack_padded_sequence(xes_a, x_lens, batch_first=True)
                        packed = True
                    except ValueError:
                        # packing failed, don't pack then
                        packed = False
                    x_encoder_output, x_hidden = self.x_rnn(xes_a)
                    if self.select_ref == 'x_output':  # use rnn output of qs
                        if packed:
                            x_encoder_output, _ = pad_packed_sequence(x_encoder_output, batch_first=True)

                        # x_encoder_output torch.Size([64, 88, 1024]) qni_es size torch.Size([64, 78, 256])
                        x_encoder_output = self.fc3(x_encoder_output)
                        attention_context_query = torch.bmm(qni_es, x_encoder_output.transpose(1, 2))  # [bsz, m, n]

                        # attention_context_query = torch.add(input=attention_context_query, value=-1e20,
                        #                                     other=torch.add(torch.neg(mask), 1))
                        # attention_context_query -= torch.add(torch.neg(mask), 1) * 1e20
                        attention_context_query -= (1 - mask) * 1e20
                        attention_context_query_score = self.softmax(attention_context_query)
                        xq = torch.bmm(attention_context_query_score, x_encoder_output)  # [bsz,m,e]
                    else:
                        # use last hidden of xs encoder to select context
                        if self.dirs > 1:
                            # take elementwise max between forward and backward hidden states
                            # NOTE: currently using max, but maybe should use Linear
                            if isinstance(self.x_rnn, nn.LSTM):
                                x_hidden = (x_hidden[0].view(-1, self.dirs, bsz, self.hsz).max(1)[0],
                                            x_hidden[1].view(-1, self.dirs, bsz, self.hsz).max(1)[0])
                            else:
                                x_hidden = x_hidden.view(-1, self.dirs, bsz, self.hsz).max(1)[0]
                        if type(x_hidden) == tuple:
                            x_hidden = self.fc3(x_hidden[0][-1])
                        else:
                            x_hidden = self.fc3(x_hidden[-1])
                        xq = x_hidden.unsqueeze(1).expand_as(qni_es)  # assert emb_size= hsz

            qni_es = self.dropout(self.fc1(torch.cat((qni_es, xq), -1)))  # bsz , n , d
            # zero out the padding idx shift bubble to right and calculalte lengths again
            if self.rl:  # use rl to select qni according to qs  and then concat selected_qni with qes
                # print('prev xs[-1]', xs[-1])
                # 已验证用rl fake和普通的两个xs和xs_lens是一样的，跑的是59的最后几个
                xs = torch.cat((qni, qs), 1)
                # print('cat qni qs xs[-1]', xs[-1])
                log_p, actions, percentage = self.select_context_word(qni_es, qni_mask, start_rl)
                all_mask = torch.cat((actions, qs_mask), 1)  # bsz , (m+n)  1. generate new mask
                x_lens = torch.sum(all_mask, dim=-1)
                # print('before sorted x_lens', x_lens)
                max_len = int(torch.max(x_lens))
                # no i have choice to lt(new_xs)
                xs = torch.where(all_mask.byte(), xs, torch.zeros_like(xs))  # 2.mask out more positions
                # print('mask out more positions  xs[-1]', xs[-1])
                all_mask_ndarray = all_mask.cpu().numpy()
                xs_mask_index = torch.from_numpy(np.argsort(-all_mask_ndarray, axis=-1, kind='mergesort')).cuda()
                # >> > c = torch.tensor([[1, 8, 4, 7, 0], [3, 4, 0, 2, 5]])
                # >> > _, id = torch.sort(c, descending=True)
                # tensor([[1, 3, 2, 0, 4],
                #         [4, 1, 0, 3, 2]])
                # >> > res = torch.gather(c, 1, id)
                # tensor([[8, 7, 4, 1, 0],
                #         [5, 4, 3, 2, 0]])
                xs = torch.gather(xs, 1, xs_mask_index)  # bsz * (m+n) # 3. shift bubble xs
                # print('shift bubble xs[-1]', xs[-1])
                sorted_x_lens, x_index_sorted = torch.sort(x_lens, descending=True)  # e.g. [3,0,2,1] the 3's position
                # of the origin qs is longest
                x_invert_index = []  # e.g. [1,3,2,0]
                for i in range(int(x_lens.size()[0])):
                    for j in range(len(x_index_sorted.tolist())):
                        if x_index_sorted[j] == i:
                            x_invert_index.append(j)
                x_invert_index = torch.LongTensor(x_invert_index).cuda()
                xs = torch.index_select(xs, 0, x_index_sorted)  # 4 re sort to  rnn
                x_lens = sorted_x_lens
                # print('after rsort xs[-1]', xs[-1])
                # print('after sorted x_lens', x_lens)
                xs = torch.narrow(xs, 1, 0, max_len)  # dim start end
                xes = self.lt(xs)
                # 5.generate attn mask
                all_mask_shift_bubble, _ = torch.sort(all_mask, dim=-1, descending=True)
                attn_mask = torch.narrow(all_mask_shift_bubble, 1, 0, max_len)  # dim start end
            else:  # fuse query to xs ,and link fused xs with qs to encode # link qni_es with qes as whole context to seq2seq
                xes = torch.cat((qni_es, qes), 1)  # new cated xes is longer than origin xes
                all_mask = torch.cat((qni_mask, qs_mask), 1)  # bsz , (m+n)
                # 2.mask out encode results which is out of origin seq length
                xes = torch.where(all_mask.byte(), xes, torch.zeros_like(xes))
                all_mask_ndarray = all_mask.cpu().numpy()
                xs_mask_index = torch.from_numpy(np.argsort(-all_mask_ndarray, axis=-1, kind='mergesort')).cuda()
                xs_mask_index = xs_mask_index.unsqueeze(2).expand_as(xes)
                # 3. shift bubble xs
                xes_shift_bubble = torch.gather(xes, 1, xs_mask_index)  # bsz * (m+n),dim
                xes = xes_shift_bubble
                x_lens = torch.sum(all_mask, dim=-1)  # bsz
                log_p = None
                actions = None
                percentage = None
            try:
                xes = pack_padded_sequence(xes, x_lens, batch_first=True) # will shorten length if max(x_lens) < len(qni+qs)
                packed = True
            except ValueError:
                # packing failed, don't pack then
                packed = False
            except RuntimeError as e:
                print(e)
                # print('\norigin x_lens',x_lens_origin)
                print('\nconcat x_lens', x_lens)

            encoder_output, hidden = self.rnn(xes)

            # hidden [num_layers,bsz,dim] encoder_output [bsz,seq_len,dim]
            if packed:
                encoder_output, _ = pad_packed_sequence(encoder_output,
                                                        batch_first=True)
            if self.dirs > 1:
                # take elementwise max between forward and backward hidden states
                # NOTE: currently using max, but maybe should use Linear
                if isinstance(self.rnn, nn.LSTM):
                    hidden = (hidden[0].view(-1, self.dirs, bsz, self.hsz).max(1)[0],
                              hidden[1].view(-1, self.dirs, bsz, self.hsz).max(1)[0])
                else:
                    hidden = hidden.view(-1, self.dirs, bsz, self.hsz).max(1)[0]
        if self.rl and not (self.is_normal or(qs is None and q_invert_index is None and qni is None and qni_invert_index is None)):
            encoder_output = torch.index_select(encoder_output, 0, x_invert_index)
            if type(hidden) == tuple:
                hidden = tuple(torch.index_select(hidden_i, 1, x_invert_index) for hidden_i in hidden)
            else:
                hidden = torch.index_select(hidden, 1, x_invert_index)
        # print(' rl', self.rl)  # True
        # print(' is_normal', self.is_normal)  # False
        #  print('encoder_output size()',encoder_output.size(), 'hidden0 size',hidden[0].size(), 'attn_mask size()',attn_mask.size())
        #  encoder_output size() torch.Size([64, 98, 1024]) hidden0 size torch.Size([2, 64, 1024]) attn_mask size() torch.Size([64, 114])
        # xes size torch.Size([64, 1, 1024]) enc_out size torch.Size([64, 98, 1024])
        return encoder_output, hidden, attn_mask, log_p, actions, percentage


class Decoder(nn.Module):
    def __init__(self, num_features, padding_idx=0, rnn_class='lstm',
                 emb_size=128, hidden_size=128, num_layers=2, dropout=0.1,
                 bidir_input=False, share_output=True,
                 attn_type='none', attn_length=-1, attn_time='pre',
                 sparse=False, numsoftmax=1, softmax_layer_bias=False):
        super().__init__()

        if padding_idx != 0:
            raise RuntimeError('This module\'s output layer needs to be fixed '
                               'if you want a padding_idx other than zero.')

        self.dropout = nn.Dropout(p=dropout)
        self.layers = num_layers
        self.hsz = hidden_size
        self.esz = emb_size

        self.lt = nn.Embedding(num_features, emb_size, padding_idx=padding_idx,
                               sparse=sparse)
        self.rnn = rnn_class(emb_size, hidden_size, num_layers,
                             dropout=dropout, batch_first=True)

        # rnn output to embedding
        if hidden_size != emb_size and numsoftmax == 1:
            # self.o2e = RandomProjection(hidden_size, emb_size)
            # other option here is to learn these weights
            self.o2e = nn.Linear(hidden_size, emb_size, bias=False)
        else:
            # no need for any transformation here
            self.o2e = lambda x: x
        # embedding to scores, use custom linear to possibly share weights
        shared_weight = self.lt.weight if share_output else None
        self.e2s = Linear(emb_size, num_features, bias=softmax_layer_bias,
                          shared_weight=shared_weight)
        self.shared = shared_weight is not None

        self.attn_type = attn_type
        self.attn_time = attn_time
        self.attention = AttentionLayer(attn_type=attn_type,
                                        hidden_size=hidden_size,
                                        emb_size=emb_size,
                                        bidirectional=bidir_input,
                                        attn_length=attn_length,
                                        attn_time=attn_time)

        self.numsoftmax = numsoftmax
        if numsoftmax > 1:
            self.softmax = nn.Softmax(dim=1)
            self.prior = nn.Linear(hidden_size, numsoftmax, bias=False)
            self.latent = nn.Linear(hidden_size, numsoftmax * emb_size)
            self.activation = nn.Tanh()

    def forward(self, xs, hidden, encoder_output, attn_mask=None, topk=1):
        xes = self.dropout(self.lt(xs))
        if self.attn_time == 'pre':
            xes = self.attention(xes, hidden, encoder_output, attn_mask)
        if xes.dim() == 2:
            # if only one token inputted, sometimes needs unsquezing
            xes.unsqueeze_(1)
        output, new_hidden = self.rnn(xes, hidden)
        if self.attn_time == 'post':
            output = self.attention(output, new_hidden, encoder_output, attn_mask)

        if self.numsoftmax > 1:
            bsz = xs.size(0)
            seqlen = xs.size(1) if xs.dim() > 1 else 1
            latent = self.latent(output)
            active = self.dropout(self.activation(latent))
            logit = self.e2s(active.view(-1, self.esz))

            prior_logit = self.prior(output).view(-1, self.numsoftmax)
            prior = self.softmax(prior_logit)  # softmax over numsoftmax's

            prob = self.softmax(logit).view(bsz * seqlen, self.numsoftmax, -1)
            probs = (prob * prior.unsqueeze(2)).sum(1).view(bsz, seqlen, -1)
            scores = probs.log()
        else:
            e = self.dropout(self.o2e(output))
            scores = self.e2s(e)

        # select top scoring index, excluding the padding symbol (at idx zero)
        # we can do topk sampling from renoramlized softmax here, default topk=1 is greedy
        if topk == 1:
            _max_score, idx = scores.narrow(2, 1, scores.size(2) - 1).max(2)
        elif topk > 1:
            max_score, idx = torch.topk(F.softmax(scores.narrow(2, 1, scores.size(2) - 1), 2), topk, dim=2, sorted=False)
            probs = F.softmax(scores.narrow(2, 1, scores.size(2) - 1).gather(2, idx), 2).squeeze(1)
            dist = torch.distributions.categorical.Categorical(probs)
            samples = dist.sample()
            idx = idx.gather(-1, samples.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        preds = idx.add_(1)

        return preds, scores, new_hidden


class Ranker(object):
    def __init__(self, decoder, padding_idx=0, attn_type='none'):
        super().__init__()
        self.decoder = decoder
        self.NULL_IDX = padding_idx
        self.attn_type = attn_type

    def forward(self, cands, cand_inds, decode_params):
        start, hidden, enc_out, attn_mask = decode_params

        hid, cell = (hidden, None) if isinstance(hidden, torch.Tensor) else hidden
        if len(cand_inds) != hid.size(1):
            cand_indices = start.detach().new(cand_inds)
            hid = hid.index_select(1, cand_indices)
            if cell is None:
                hidden = hid
            else:
                cell = cell.index_select(1, cand_indices)
                hidden = (hid, cell)
            enc_out = enc_out.index_select(0, cand_indices)
            attn_mask = attn_mask.index_select(0, cand_indices)

        cand_scores = []

        for i in range(len(cands)):
            curr_cs = cands[i]

            n_cs = curr_cs.size(0)
            starts = start.expand(n_cs).unsqueeze(1)
            scores = 0
            seqlens = 0
            # select just the one hidden state
            if isinstance(hidden, torch.Tensor):
                nl = hidden.size(0)
                hsz = hidden.size(-1)
                cur_hid = hidden.select(1, i).unsqueeze(1).expand(nl, n_cs, hsz)
            else:
                nl = hidden[0].size(0)
                hsz = hidden[0].size(-1)
                cur_hid = (hidden[0].select(1, i).unsqueeze(1).expand(nl, n_cs, hsz).contiguous(),
                           hidden[1].select(1, i).unsqueeze(1).expand(nl, n_cs, hsz).contiguous())

            cur_enc, cur_mask = None, None
            if attn_mask is not None:
                cur_mask = attn_mask[i].unsqueeze(0).expand(n_cs, attn_mask.size(-1))
                cur_enc = enc_out[i].unsqueeze(0).expand(n_cs, enc_out.size(1), hsz)
            # this is pretty much copied from the training forward above
            if curr_cs.size(1) > 1:
                c_in = curr_cs.narrow(1, 0, curr_cs.size(1) - 1)
                xs = torch.cat([starts, c_in], 1)
            else:
                xs, c_in = starts, curr_cs
            if self.attn_type == 'none':
                preds, score, cur_hid = self.decoder(xs, cur_hid, cur_enc, cur_mask)
                true_score = F.log_softmax(score, dim=2).gather(
                    2, curr_cs.unsqueeze(2))
                nonzero = curr_cs.ne(0).float()
                scores = (true_score.squeeze(2) * nonzero).sum(1)
                seqlens = nonzero.sum(1)
            else:
                for i in range(curr_cs.size(1)):
                    xi = xs.select(1, i)
                    ci = curr_cs.select(1, i)
                    preds, score, cur_hid = self.decoder(xi, cur_hid, cur_enc, cur_mask)
                    true_score = F.log_softmax(score, dim=2).gather(
                        2, ci.unsqueeze(1).unsqueeze(2))
                    nonzero = ci.ne(0).float()
                    scores += true_score.squeeze(2).squeeze(1) * nonzero
                    seqlens += nonzero

            scores /= seqlens  # **len_penalty?
            cand_scores.append(scores)

        max_len = max(len(c) for c in cand_scores)
        cand_scores = torch.cat([pad(c, max_len).unsqueeze(0) for c in cand_scores], 0)
        preds = cand_scores.sort(1, True)[1]
        return preds, cand_scores


class Linear(nn.Module):
    """Custom Linear layer which allows for sharing weights (e.g. with an
    nn.Embedding layer).
    """
    def __init__(self, in_features, out_features, bias=True,
                 shared_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared = shared_weight is not None

        # init weight
        if not self.shared:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            if (shared_weight.size(0) != out_features or
                    shared_weight.size(1) != in_features):
                raise RuntimeError('wrong dimensions for shared weights')
            self.weight = shared_weight

        # init bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if not self.shared:
            # weight is shared so don't overwrite it
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = self.weight
        if self.shared:
            # detach weight to prevent gradients from changing weight
            # (but need to detach every time so weights are up to date)
            weight = weight.detach()
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class RandomProjection(nn.Module):
    """Randomly project input to different dimensionality."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features),
                                requires_grad=False)  # fix weights
        self.reset_parameters()

    def reset_parameters(self):
        # experimentally: std=1 appears to affect scale too much, so using 0.1
        self.weight.data.normal_(std=0.1)
        # other init option: set randomly to 1 or -1
        # self.weight.data.bernoulli_(self.weight.fill_(0.5)).mul_(2).sub_(1)

    def forward(self, input):
        return F.linear(input, self.weight)


class AttentionLayer(nn.Module):
    def __init__(self, attn_type, hidden_size, emb_size, bidirectional=False,
                 attn_length=-1, attn_time='pre'):
        super().__init__()
        self.attention = attn_type

        if self.attention != 'none':
            hsz = hidden_size
            hszXdirs = hsz * (2 if bidirectional else 1)
            if attn_time == 'pre':
                # attention happens on the input embeddings
                input_dim = emb_size
            elif attn_time == 'post':
                # attention happens on the output of the rnn
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')
            self.attn_combine = nn.Linear(hszXdirs + input_dim, input_dim,
                                          bias=False)

            if self.attention == 'local':
                # local attention over fixed set of output states
                if attn_length < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = attn_length
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz + input_dim, attn_length, bias=False)
                # combines attention weights with encoder outputs
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz, bias=False)
                self.attn_v = nn.Linear(hsz, 1, bias=False)
            elif self.attention == 'general':
                # equivalent to dot if attn is identity
                self.attn = nn.Linear(hsz, hszXdirs, bias=False)

    def forward(self, xes, hidden, enc_out, attn_mask=None):
        if self.attention == 'none':
            return xes

        if type(hidden) == tuple:
            # for lstms use the "hidden" state not the cell state
            hidden = hidden[0]
        last_hidden = hidden[-1]  # select hidden state from last RNN layer

        if self.attention == 'local':
            if enc_out.size(1) > self.max_length:
                offset = enc_out.size(1) - self.max_length
                enc_out = enc_out.narrow(1, offset, self.max_length)
            h_merged = torch.cat((xes.squeeze(1), last_hidden), 1)
            attn_weights = F.softmax(self.attn(h_merged), dim=1)
            if attn_weights.size(1) > enc_out.size(1):
                attn_weights = attn_weights.narrow(1, 0, enc_out.size(1))
        else:
            hid = last_hidden.unsqueeze(1)
            if self.attention == 'concat':
                hid = hid.expand(last_hidden.size(0),
                                 enc_out.size(1),
                                 last_hidden.size(1))
                h_merged = torch.cat((enc_out, hid), 2)
                active = F.tanh(self.attn(h_merged))
                attn_w_premask = self.attn_v(active).squeeze(2)
            elif self.attention == 'dot':
                if hid.size(2) != enc_out.size(2):
                    # enc_out has two directions, so double hid
                    hid = torch.cat([hid, hid], 2)
                attn_w_premask = (
                    torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1))
            elif self.attention == 'general':
                hid = self.attn(hid)
                attn_w_premask = (
                    torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1))
            # calculate activation scores
            if attn_mask is not None:
                # remove activation from NULL symbols
                attn_w_premask -= (1 - attn_mask) * 1e20
            attn_weights = F.softmax(attn_w_premask, dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        output = F.tanh(self.attn_combine(merged).unsqueeze(1))
        return output
