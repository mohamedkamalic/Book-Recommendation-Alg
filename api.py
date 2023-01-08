import flask
import numpy as np
from flask import request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
app = flask.Flask(__name__)
app.config["DEBUG"] = True


books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]

@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route('/api/v1/resources/books/all', methods=['GET'])
def api_all():
    return jsonify(books)


@app.route('/api/v1/resources/books', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for book in books:
        if book['id'] == id:
            results.append(book)

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)

#start machine learning algo

@app.route('/api/get_rec', methods=['GET'])
def api_rec():
    
    if 'books' in request.args:
        query = eval(request.args['books'])
        query_list=list(query.values())
        test_array=np.array(query_list)
        #now you have a requseted book as np array
        books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
        books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
        users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
        users.columns = ['userID', 'Location', 'Age']
        ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
        ratings.columns = ['userID', 'ISBN', 'bookRating']
        plt.rc("font", size=15)
        ratings.bookRating.value_counts(sort=False).plot(kind='bar')
        plt.title('Rating Distribution\n')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        #plt.savefig('system1.png', bbox_inches='tight')
        #plt.show()
        users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
        plt.title('Age Distribution\n')
        plt.xlabel('Age')
        plt.ylabel('Count')
        #plt.savefig('system2.png', bbox_inches='tight')
        #plt.show()
        counts1 = ratings['userID'].value_counts()
        ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 100].index)]
        counts = ratings['bookRating'].value_counts()
        ratings = ratings[ratings['bookRating'].isin(counts[counts >= 50].index)]
        combine_book_rating = pd.merge(ratings, books, on='ISBN')
        columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
        combine_book_rating = combine_book_rating.drop(columns, axis=1)
        combine_book_rating.head()
        combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])
        
        book_ratingCount = (combine_book_rating.
             groupby(by = ['bookTitle'])['bookRating'].
             count().
             reset_index().
             rename(columns = {'bookRating': 'totalRatingCount'})
             [['bookTitle', 'totalRatingCount']]
            )
        book_ratingCount.head()
        rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
        rating_with_totalRatingCount.head()
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        popularity_threshold =5
        rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
        rating_popular_book.head()
        rating_popular_book.shape
        combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')
        from scipy.sparse import csr_matrix
        us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada|india|egypt")]
        us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
        us_canada_user_rating.head()
        us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
        us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
        us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)
        #print(us_canada_user_rating_matrix)
        
        
        from sklearn.neighbors import NearestNeighbors
        respond =np.array(300)
        query_index =0
        #getting recommendition for each book
        for j in range(0,len(test_array)):
            query_index = test_array[j]
            model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
            model_knn.fit(us_canada_user_rating_matrix)
            us_canada_user_rating_pivot.iloc[0,:].values.reshape(1,-1)
            distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.loc[query_index,:].values.reshape(1, -1), n_neighbors = 25)
            us_canada_user_rating_pivot.loc[query_index]
            for i in range(0, len(distances.flatten())):    
                respond =np.append(respond,indices)
                
        respond = np.unique(respond, axis=0)
        #convert respond list to dictinary of books
        new_dec = {'BOOK_'+str(i):us_canada_user_rating_pivot.index[respond.flatten()[i]]  for i in range(0, len(respond), 1)}
       
    else:
        return "Error: No id field provided. Please specify an id."
    
    #query = {'book'+str(i): query_list[i] for i in range(0, 2, 1)}
    return jsonify(new_dec)
    #return jsonify(query)
@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

app.run()

