import pandas as pd
import graphlab as gl
from flask import Flask, g, render_template, request


app = Flask(__name__)


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/recommend", methods=['POST'])
def recommend():
    input_boxes = ['beer_input1_id', 'beer_input2_id', 'beer_input3_id']
    user_input = []
    for box in input_boxes:
        submission = request.form[box]
        if submission != '':
            submission = int(submission)
            user_input.append(submission)

    user_input = pd.DataFrame({'user_id': [20000 for _ in user_input], 'item_id': user_input})
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
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == '__main__':
    main()
