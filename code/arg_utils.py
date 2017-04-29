''' Utilities for command-line arguments. '''

def add_common_arguments(parser):
  ''' Add the arguments. '''
  parser.add_argument(
    '-clf', '--classifiers-to-be-used', default='RandomForestClassifier', type=str,
    help='The classifier to be used in analysis')
  parser.add_argument(
    '-name', '--bacteria-name',
    default='Staphylococcus_methicillin', type=str, help='')