from sklearn.linear_model import SGDRegressor
from azureml.core import Workspace
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
import argparse
import numpy as np
import pandas as pd

def main():
    #Get data
    #ws = Workspace.get(name='quick-starts-ws-187058',resource_group='aml-quickstarts-187058',subscription_id='2c48c51c-bd47-40d4-abbe-fb8eabd19c8c')
    
    #df_bonds_tab = ws.datasets['df_bonds']
    #df_bonds= df_bonds_tab.to_pandas_dataframe()
    df_bonds=pd.read_csv('input_df.csv')
    x=df_bonds.drop('OAS',1)
    y=df_bonds['OAS']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10)
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates")
    parser.add_argument('--l1_ratio', type=float, default=1.0,
                    help="The elastic Net mising parameter")
    parser.add_argument('--max_iter', type=int, default=70, help="Maximum number of iterations to converge")
    args, unknown = parser.parse_known_args()
    run = Run.get_context()
    run.log("Regularization Strength:", np.float(args.alpha))
    run.log("Max iterations:", np.int(args.max_iter))
    model = SGDRegressor(penalty='elasticnet',alpha=args.alpha,l1_ratio=args.l1_ratio, max_iter=args.max_iter).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()