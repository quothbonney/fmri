import torch
import random

def validate_context_and_fix(c_ix, t_ix, batch_size):
  for i in range(batch_size):
    for context in c_ix[i]:
      if torch.equal(context, t_ix[i]):
        dim = random.choice([0, 1])
        shift = random.choice([-1, 1])
        context[dim] += shift
  return c_ix

def validate_context(c_ix, t_ix, batch_size):
  for i in range(batch_size):
    for context in c_ix[i]:
      if torch.equal(context, t_ix[i]):
        return False
  return True
