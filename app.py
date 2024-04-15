from flask import Flask, render_template,request,url_for,redirect
import pickle
import numpy as np

popular_df =pickle.load(open('popular.pkl','rb'))
pt =pickle.load(open('pt.pkl','rb'))
products =pickle.load(open('products.pkl','rb'))
similarity_score =pickle.load(open('similarity_score.pkl','rb'))
app= Flask(__name__)

@app.route('/')

def index():
     return render_template('index.html',
                            product_name=list(popular_df['product_name'].values),
                            price=list(popular_df['actual_price'].values),
                            rating=list(popular_df['rating'].values),
                            img_link=list(popular_df['img_link'].values),
                            )

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    matching_products = [product for product in pt.index if user_input.lower() in product.lower()]
    
    data = []

    if matching_products:
        product_name = matching_products[0]  # Selecting the first matched product
        index = pt.index.get_loc(product_name)
        similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:5]
        print("Similar products for '{}':".format(product_name))
        
        for i, (product_index, score) in enumerate(similar_items, 1):
            item = []
            temp_df = products[products['product_name'] == pt.index[product_index]]
            item.append(temp_df.drop_duplicates('product_name')['product_name'].values[0])
            item.append(temp_df.drop_duplicates('product_name')['rating'].values[0])
            item.append(temp_df.drop_duplicates('product_name')['actual_price'].values[0])
            item.append(temp_df.drop_duplicates('product_name')['img_link'].values[0])
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