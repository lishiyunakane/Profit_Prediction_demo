from flask import Flask, request, render_template
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import openpyxl

app = Flask(__name__)


def generate_output(file):
    df = pd.read_excel(file, skiprows=6, header=0)
    new_df = df
    new_df['Project No'] = new_df['Project No'].str[0]

    def get_country(rate):
        if rate == 1.0:
            return 'S'
        else:
            return 'O'

    new_df['Country'] = new_df['Exch\nRate'].apply(get_country)
    new_df['Target\nGross Profit (%)'] = new_df['Target\nGross Profit (%)'] *100
    new_df['Sales\nAmount\n(Home)'] = new_df['Sales\nAmount\n(Home)'].str.replace(',', '').astype(float)
    df1 = new_df[['Project No', 'Sales\nAmount\n(Home)', 'Target\nGross Profit (%)', 'Country']]

    dummies = pd.get_dummies(new_df['Client'], prefix='Client')
    dummies2 = pd.get_dummies(new_df['Commencial\nI/C'])
    df1 = pd.concat([df1, dummies], axis=1)
    df1 = pd.concat([df1, dummies2], axis=1)
    dummies3 = pd.get_dummies(df1[['Project No', 'Country']])
    df1 = pd.concat([df1.drop(['Project No', 'Country'], axis=1), dummies3], axis=1)

    df_empty = pd.read_excel('s1.xlsx')
    df_empty[['Sales Amount (Home)','Target Gross Profit (%)']] = df1.loc[:,['Sales\nAmount\n(Home)','Target\nGross Profit (%)']]
    df_empty.fillna(0, inplace=True)
    for col in df1.columns:
        if col in df_empty.columns:
            for index, row in df1.iterrows():
                if row[col] == 1:
                    df_empty.loc[index, col] = 1

    df_empty = df_empty.drop('Actual Gross Profit (%)', axis=1)
    model = load_model('my_model_weights.h5')
    data = df_empty

    scaler = StandardScaler()
    data1 = pd.read_csv('output.csv')
    target = data1['Actual Gross Profit (%)']
    features = data1.drop('Actual Gross Profit (%)', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    new_data_scaled = scaler.transform(data)

    predictions = model.predict(new_data_scaled)
    df = pd.read_excel(file, skiprows=6, header=0)
    project_name = df[['Project No','Description','Client','Sales\nAmount\n(Home)']]
    df2 = pd.DataFrame(predictions)
    df2 = df2.applymap(lambda x: round(x, 3))
    results = pd.concat([project_name, df2], axis=1)

    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        output = generate_output("test.xlsx")  # Passing the file name directly here
        return render_template('result.html', output=output)
    return render_template('index.html')


if __name__ == '__main__':

    app.run(debug=True)

