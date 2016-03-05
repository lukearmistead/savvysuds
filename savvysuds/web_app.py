import pandas as pd
import graphlab as gl
from flask import Flask
from flask import render_template
from flask import Flask, request
from flask import g
from flask import jsonify
app = Flask(__name__)

'''
todo
- add autocomplete to find beers you want
- replace weird characters in beer names
- look into sudo command to get gl running
- clean up ui
'''

@app.route("/")
def main():
    return render_template('index.html')

# @app.route("/suggest")
# def suggest():
#     user_input = request.args.get('input')
#     return jsonify({'asdf': 'you typed ' + user_input})
    # implement python search of database
    # return jsonified list of records

@app.route("/recommend", methods=['POST'])
def recommend():
    brew1 = int(request.form['beer_input1_id'])
    brew2 = int(request.form['beer_input2_id'])
    brew3 = int(request.form['beer_input3_id'])

    user_input = [brew1, brew2, brew3]

    user_input = pd.DataFrame({'a': [20000 for _ in user_input], 'b': user_input})
    user_input.columns = ['user_id', 'item_id']
    user_input = gl.SFrame(user_input)
    model = gl.load_model('../models/item_similarity_model')
    pred = list(model.recommend(users=[20000], k=5, new_observation_data=user_input, diversity=3)['item_id'])

    beers = pd.read_csv('../data/raw_data/beers_3.csv')
    beer_recs = beers[beers['id'].isin(pred)]
    beer_recs = beer_recs[['name', 'brewery_name', 'style', 'score']]
    beer_recs.columns = ['brew', 'brewery', 'style', 'untappd score']
    beer_recs = beer_recs.to_html(columns=['brew', 'brewery', 'style', 'untappd score'],
                                  index=False)

    beer_recs = beer_recs.replace('border="1" class="dataframe"',
                                  'class=table table-hover')
    return render_template('index.html', recommend=True, beer_recs=beer_recs)


def main():
    # with app.app_context():
        # app.g.beers = pd.read_csv('../data/raw_data/beers.csv')
    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == "__main__":
    main()
