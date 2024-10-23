import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

path = '/Users/cadepreister/Desktop/Intro to Python Data Analytics/'
sep = '-'*80

#Question one parts
def one_p1():
    run_num = 0
    results = []

    while True:
        print(sep)
        sheet = input("Enter a sheet number (within the range of 1 - 4) to examine, or q to quit: ").strip()
        if sheet.lower() == 'q':
            break
        resp_var = input("Select a column for your response variable (0 - 3): ").strip()
        user_test = input("Select a ratio setting (real number between 0 - 0.98): ").strip()

        try:
            int(sheet)
            int(resp_var)
            float(user_test)

            if 0 >= int(sheet) or int(sheet) >= 5:
                raise ValueError
            if -1 >= int(resp_var) or int(resp_var) >= 4:
                raise ValueError
            if 0 > float(user_test) or float(user_test) > 0.99:
                raise ValueError

        except ValueError:
            print("Enter a number between 1 & 4 (sheet #)"
                  " | 0 & 3 (response variable)"
                  " | 0 & 0.98 (ratio setting)")

        else:
            run_num += 1
            results.append(one_p2(run_num, sheet, user_test, resp_var))

    if results:
        one_p3(results)


def one_p2(run_num, sheet, user_test, resp_var):
    print(sep)
    ef = pd.read_excel(f'{path}iris.xlsx', sheet_name=f'Sheet{sheet}')
    print(f"Ratio setting: {user_test}"
          f"\nLength (in rows): {ef.shape[0]}"
          f"\nTraining Columns (explanatory variables): 0 - 3")

    y = np.array(ef[ef.columns[int(resp_var)]])
    print(f'Target estimation (response variable) data in column: '
          f'{resp_var} - The {ef.columns[int(resp_var)]}')

    if resp_var == '0':
        x = np.array(ef[ef.columns[[1, 2, 3]]])
    elif resp_var == '1':
        x = np.array(ef[ef.columns[[0, 2, 3]]])
    elif resp_var == '2':
        x = np.array(ef[ef.columns[[0, 1, 3]]])
    elif resp_var == '3':
        x = np.array(ef[ef.columns[[0, 1, 2]]])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(user_test))

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    y_predict = lr.predict(x_test)
    print("\ty_test\ty_predict")
    for i in range(0, len(y_predict), 1):
        print(f"{i}\t{y_test[i]}\t\t{y_predict[i]:.2f}")
    r_squared = lr.score(x_test, y_test)
    print('\nR-squared: %.4f' % r_squared)

    first_cols = ef.iloc[:, :4]
    corr_table = first_cols.corr()
    if sheet == '1':
        curr_species = 'Overall Species'
    else:
        curr_species = ef.iloc[1, 4]
    print(f"{sep}\n\tCorrelation Coefficient of {curr_species} - Sheet{sheet}\n{corr_table}")

    cols = '0, 1, 2, 3'

    run_dict = {
        'Run #' : run_num,
        'Sheet #' : sheet,
        'Training Columns' : cols.replace(f'{resp_var}, ', '').replace(f', {resp_var}', ''),
        'Training ratio setting' : float(user_test)*100,
        'R-squared score' : r_squared
    }

    return run_dict


def one_p3(r_square_data):
    usr_file = input("What would you like the name of your file to be (new_dataset.xlsx): ")
    if not usr_file.endswith('.xlsx'):
        usr_file += '.xlsx'

    r_square_df = pd.DataFrame(r_square_data)
    with pd.ExcelWriter(f'{path}{usr_file}') as writer:
        r_square_df.to_excel(writer, sheet_name='Sheet1')
    print(f"The dataset has been saved to {usr_file}!\n{sep}")

    r2_vals = r_square_df['R-squared score']
    r2_max = r2_vals.max()
    r2_min = r2_vals.min()
    r2_mean = r2_vals.mean()
    r2_variance = r2_vals.var()
    print(f"Maxima: {r2_max:.4f}\nMinima: {r2_min:.4f}\nMean: {r2_mean:.4f}\nVariance: {r2_variance:.4f}\n{sep}")

#Question two parts
def two_p1():
    run_num = 0
    results = []

    print(f"It is now time to analyze the overall species data (Sheet 1)!")

    while True:
        print(sep)
        user_test = input("Select a ratio setting (real number between 0 - 0.98) or q to quit: ").strip()
        if user_test.lower() == 'q':
            break

        try:
            float(user_test)

            if 0 > float(user_test) or float(user_test) >= 0.99:
                raise ValueError

        except ValueError:
            print("Enter a number between 0 & 0.98!")

        else:
            run_num += 1
            results.append(two_p2(run_num, user_test))

    if results:
        one_p3(results)


def two_p2(run_num, user_test):
    print(sep)
    ef = pd.read_excel(f'{path}iris.xlsx', sheet_name=f'Sheet1')
    print(f"Ratio setting: {user_test}"
          f"\nLength (in rows): {ef.shape[0]}"
          f"\nColumns for features (explanatory variables): 0 - 3")

    y = np.array(ef[ef.columns[4]])
    x = np.array(ef[ef.columns[0:3]])
    print(f'Target estimation (label) data in column: '
          f'4 - The Overall Species')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(user_test))

    lr = LogisticRegression(max_iter=200)
    lr.fit(x_train, y_train)

    y_predict = lr.predict(x_test)
    print(f"\ty_test{'\t'*6}y_predict")
    for i in range(0, len(y_predict), 1):
        print(f"{i}\t{y_test[i]:<20}\t\t{y_predict[i].ljust(30):<20}")
    r_squared = lr.score(x_test, y_test)
    print('\nR-squared: %.4f' % r_squared)

    run_dict ={
        'Run #': run_num,
        'Training Columns': '0, 1, 2, 4',
        'Training ratio setting': float(user_test) * 100,
        'R-squared score': r_squared
    }

    return run_dict

def main():
    one_p1()
    two_p1()


#Main function
if __name__ == "__main__":

    main()
