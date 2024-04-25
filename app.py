from flask import Flask, render_template,request,url_for,redirect
import pickle
import numpy as np

popular_df =pickle.load(open('popular.pkl','rb'))
# pt =pickle.load(open('pt.pkl','rb'))
products =pickle.load(open('products.pkl','rb'))
similarity_score =pickle.load(open('similarity_score.pkl','rb'))
app= Flask(__name__)

@app.route('/')

def index():
     return render_template('index.html',
                            product_name=list(popular_df['ProductTitle'].values),
                            
                            img_link=list(popular_df['ImageURL'].values),
                            )

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    matches = products[products['name'].str.contains(user_input, case=False)]
    data = []

    if not matches.empty:
        searched_index = matches.index[0]  # Get the index of the first match
        similar_items_all = []
        for index in matches.index:
            similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
            similar_items_all.extend(similar_items)
        similar_items_all = sorted(similar_items_all, key=lambda x: x[1], reverse=True)

        recommended_indices = set()  # To keep track of recommended product indices
        recommended_count = 0

        for product_index, score in similar_items_all:
            if product_index != searched_index and product_index not in recommended_indices:
                # Add the product index to the set of recommended indices
                recommended_indices.add(product_index)
                recommended_count += 1
                if recommended_count > 5:
                    break  # Stop when 5 unique recommendations are found
                item = []
                temp_df = products.iloc[product_index]
                item.append(temp_df['ProductTitle'])

                item.append(temp_df['ImageURL'])
                data.append(item)
    else:
        print("No products found matching the specified partial name.")

    print( data)

    return render_template('recommend.html',data=data)



@app.route('/recommend_image')
def redirect_to_streamlit():
    streamlit_url = 'http://localhost:8501'
    return redirect(streamlit_url)


if __name__ == '__main__':
    app.run(debug=True)