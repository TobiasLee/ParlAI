#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Inheritence around add_cmdline_args can be tricky. This serves as an
example, and verifies inheritence is behaving correctly.
"""

import unittest
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import TorchGeneratorAgent


class FakeDict(object):
    @classmethod
    def add_cmdline_args(cls, parser):
        parser.add_argument('--dictarg', default='d')


class SubClassA(TorchGeneratorAgent):
    @classmethod
    def dictionary_class(cls):
        return FakeDict

    @classmethod
    def add_cmdline_args(cls, parser):
        parser.add_argument('--withclassinheritence', default='a')
        super(SubClassA, cls).add_cmdline_args(parser)


class SubClassB(SubClassA):
    @classmethod
    def add_cmdline_args(cls, parser):
        parser.add_argument('--withoutclassinheritence', default='b')


class TestInheritence(unittest.TestCase):
    def test_subclassA(self):
        """Verify that class A does contain the super args."""
        parser = ParlaiParser(add_model_args=True)
        opt = parser.parse_args(
            args=['--model', 'tests.test_ta_inheritence:SubClassA'], print_args=False
        )
        self.assertEqual('a', opt.get('withclassinheritence'))
        # make sure we have the dictionary arg
        self.assertEqual('d', opt.get('dictarg'))
        # something that torch agent has
        self.assertIn('no_cuda', opt)
        # something torch generator agent has
        self.assertIn('beam_size', opt)

    def test_subclassB(self):
        """Verify that class B does not contain the super args."""
        parser = ParlaiParser(add_model_args=True)
        opt = parser.parse_args(
            args=['--model', 'tests.test_ta_inheritence:SubClassB'], print_args=False
        )
        self.assertEqual('b', opt.get('withoutclassinheritence'))
        # make sure we don't have the dictionary now
        self.assertNotIn('dictarg', opt)
        # something the parent has
        self.assertNotIn('withclassinheritence', opt)
        # something that torch agent has
        self.assertNotIn('no_cuda', opt)
        # something torch generator agent has
        self.assertNotIn('beam_size', opt)


if __name__ == '__main__':
    unittest.main()
