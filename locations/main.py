import preprocessor
import processor
import analytics
from config import *

def main():
    # clean_df = preprocessor.preprocess_data()
    # labeled_visits = processor.process_data(clean_df, WEEKS)
    analytics.categorize_data(WEEKS)

main()