import argparse
import arg_utils
import file_utils



def main(args):
  """Runs!"""
  files = file_utils.make_file_names(args.bacteria_name)
  ret = file_utils.get_labels('StaphylococcusMethicillin.log')
  print ret

  
def parse_args():
  """ Parses command-line argument. """
  parser = argparse.ArgumentParser(
    description='Get ready!')
  arg_utils.add_common_arguments(parser)
  return parser.parse_args()

if __name__ == '__main__':
  main(parse_args())
