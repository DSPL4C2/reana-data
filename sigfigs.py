from decimal import Decimal
from math import log10, floor
from utils import isnan

def format_value_error(value, error):
  if isnan(value):
    return '--'
  if error == 0.0:
    return '{}'.format(value)
  (truncated_error, sigfigs) = process_error(error)
  formatted_value = format_value(value, sigfigs)
  # return '{:.2f} ± {:.2f}'.format(formatted_value, truncated_error)
  # return '{} ± {}'.format(formatted_value, truncated_error)
  return '{} ({})'.format(formatted_value, truncated_error)

def process_error(error):
  # TODO
  sigfigs = -int(floor(log10(abs(error))))
  new_error = round(error, sigfigs)
  return (new_error, sigfigs)

def format_value(value, sigfigs):
  # TODO
  return round(value, sigfigs)