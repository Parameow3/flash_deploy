import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
import random

# Load data
data = pd.read_csv("data/rating_complete.csv")
podcast_data = pd.read_csv("data/anime.csv")

# Create a user-item matrix in sparse format
user_podcast_matrix = csr_matrix((data['rating'], (data['user_id'], data['anime_id'])))


class Recommender:
    def __init__(self, user_podcast_matrix, n_neighbors=10):
        self.user_podcast_matrix = user_podcast_matrix
        self.model = self.fit_knn(n_neighbors)

    def fit_knn(self, n_neighbors):
        model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        model.fit(self.user_podcast_matrix)
        return model

    def recommend_podcasts(self, user_id, num_recommendations=5):
        if user_id >= self.user_podcast_matrix.shape[0]:
            return []

        distances, indices = self.model.kneighbors(self.user_podcast_matrix[user_id],
                                                   n_neighbors=num_recommendations + 1)
        similar_users = indices.flatten()[1:]

        user_ratings = self.user_podcast_matrix.getrow(user_id).toarray().ravel()
        rated_podcasts = user_ratings.nonzero()[0]

        recommendations = {}
        for similar_user in similar_users:
            similar_user_ratings = self.user_podcast_matrix.getrow(similar_user).toarray().ravel()
            for podcast_id, rating in enumerate(similar_user_ratings):
                if podcast_id not in rated_podcasts:
                    if podcast_id not in recommendations:
                        recommendations[podcast_id] = 0
                    recommendations[podcast_id] += rating

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_podcast_ids = [podcast_id for podcast_id, _ in sorted_recommendations[:num_recommendations]]

        recommended_podcasts = podcast_data[podcast_data['anime_id'].isin(recommended_podcast_ids)]

        return recommended_podcasts[
            ['anime_id', 'Name', 'Genres', 'Type', 'Aired', 'Producers', 'Studios', 'Source', 'Duration',
             'Rating']].to_dict(orient='records')

    def get_liked_podcasts(self, user_id):
        if user_id >= self.user_podcast_matrix.shape[0]:
            return []

        user_ratings = self.user_podcast_matrix.getrow(user_id).toarray().ravel()
        liked_podcasts = user_ratings.nonzero()[0]

        liked_podcast_details = podcast_data[podcast_data['anime_id'].isin(liked_podcasts)]
        return liked_podcast_details[
            ['anime_id', 'Name', 'Genres', 'Type', 'Aired', 'Producers', 'Studios', 'Source', 'Duration',
             'Rating']].to_dict(orient='records')


app = Flask(__name__)
recommender = Recommender(user_podcast_matrix)


# Helper function to filter and paginate data
def filter_and_paginate_data(filtered_df, page, per_page):
    total = filtered_df.shape[0]
    start = (page - 1) * per_page
    end = start + per_page
    paginated_df = filtered_df.iloc[start:end]
    return paginated_df, total


# Get recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    num_recommendations = int(request.args.get('num_recommendations', 5))

    if user_id is None:
        return jsonify({'error': 'User ID is required'}), 400

    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({'error': 'Invalid User ID'}), 400

    recommendations = recommender.recommend_podcasts(user_id, num_recommendations)

    return jsonify({'User_ID': user_id, 'recommendations': recommendations})


# Get liked podcasts
@app.route('/liked_podcasts', methods=['GET'])
def liked_podcasts():
    user_id = request.args.get('user_id')

    if user_id is None:
        return jsonify({'error': 'User ID is required'}), 400

    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({'error': 'Invalid User ID'}), 400

    liked_podcasts = recommender.get_liked_podcasts(user_id)

    return jsonify({'User_ID': user_id, 'liked_podcasts': liked_podcasts})


# Get podcasts by studios with optional randomization
@app.route('/podcasts/studios/<string:studio_name>', methods=['GET'])
def get_podcasts_by_studio(studio_name):
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    randomize = request.args.get('random', 'false').lower() == 'true'

    filtered_df = podcast_data[podcast_data['Studios'].astype(str).str.contains(studio_name, case=False, na=False)]

    if randomize:
        filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

    paginated_df, total = filter_and_paginate_data(filtered_df, page, per_page)

    result = paginated_df.to_dict(orient='records')

    response = {
        'page': page,
        'per_page': per_page,
        'total': total,
        'data': result
    }

    return jsonify(response)


# Get podcasts with optional randomization
@app.route('/podcasts', methods=['GET'])
def get_podcasts():
    query_params = request.args.to_dict()

    page = int(query_params.pop('page', 1))
    per_page = int(query_params.pop('per_page', 10))
    randomize = query_params.pop('random', 'false').lower() == 'true'

    filtered_df = podcast_data
    for key, value in query_params.items():
        if key in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[key].astype(str).str.contains(value, case=False, na=False)]

    if randomize:
        filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

    paginated_df, total = filter_and_paginate_data(filtered_df, page, per_page)

    result = paginated_df.to_dict(orient='records')

    return jsonify(result)


# Get podcasts by genres with optional randomization
@app.route('/podcasts/genres/<string:genre_name>', methods=['GET'])
def get_podcasts_by_genre(genre_name):
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    randomize = request.args.get('random', 'false').lower() == 'true'

    filtered_df = podcast_data[podcast_data['Genres'].astype(str).str.contains(genre_name, case=False, na=False)]

    if randomize:
        filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

    paginated_df, total = filter_and_paginate_data(filtered_df, page, per_page)

    result = paginated_df.to_dict(orient='records')

    response = {
        'page': page,
        'per_page': per_page,
        'total': total,
        'data': result
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
