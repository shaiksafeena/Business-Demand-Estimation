from flask import *
import csv,os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from flask_pymongo import pymongo
import re

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

app = Flask(__name__)
app.secret_key = 'forseer'


con_string = "mongodb+srv://Pragna_2803:aprv10092835@cluster0.l1mtumg.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(con_string)
db = client.get_database('safeena')
user_collection = pymongo.collection.Collection(db, 'neelima')
print("MongoDB connected Successfully")

#Index
@app.route("/")
def index():
    return render_template('index.html')

#SignIn
@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = user_collection.find_one({'email': email})
        print(user)

        if not user:
            flash('Email not Registered', 'error')
            return redirect(url_for('signin'))
        
        # Check if password is correct
        if user['password']!=password:
            flash('Password does not Match', 'error')
            return redirect(url_for('signin'))
        
        # Log user in
        session.clear()
        session['user_id'] = str(user['_id'])
        session['username'] = user['username']
        return redirect(url_for('dash_form'))
    
        
    return render_template('signin.html')


#SignUp
@app.route("/signup" , methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
         # Check if user already exists
        if user_collection.find_one({'email': email}):
            flash('User Already Registered',  'error')
            return redirect(url_for('signup'))
        
        # Insert user into database
        user = {
            'username': username,
            'email': email,
            'password': password
        }
        user_collection.insert_one(user)
        session['username'] = username
    
        return redirect(url_for('dash_form'))

    return render_template('signup.html')




#Dashboard FormPage
@app.route('/form', methods=['GET','POST'])
def dash_form():
    username = session.get('username', None)
    if not username:
        return redirect(url_for('signup'))
    return render_template('dash-form.html', username=username)


#Dashboard Further Information Page
@app.route('/info', methods=['POST','GET'])
def dash_info():
    if request.method == 'POST':
        Battery_Life = request.form.get('Battery_Life')
        Processor_Speeds = request.form.get('Processor_Speeds')
        Ram = request.form.get('Ram')
        Screen_Size = request.form.get('Screen_Size')
        Integrated_Wireless = request.form.get('Integrated_Wireless')
        Bundled_Applications = request.form.get('Bundled_Applications')
        Retail_Price = request.form['Retail_Price']
        HD_Size = request.form['HD_Size']
        Sales = 0
        fieldnames = ['Battery_Life','Bundled_Applications','HD_Size','Integrated_Wireless','Processor_Speeds','Ram','Retail_Price','Screen_Size','Sales']
        values = [Battery_Life, Bundled_Applications, HD_Size, Integrated_Wireless, Processor_Speeds, Ram, Retail_Price, Screen_Size,Sales]
        #values = [a,b,c,d,e,f,g,h]
        with open('Newdata.csv', 'wt', newline="") as inFile:
            writer = csv.writer(inFile, delimiter=',')
            writer.writerow(i for i in fieldnames)

            writer.writerow(j for j in values)
        #with open('test11.csv','w', newline="") as inFile:
         #   writer = csv.DictWriter(inFile, fieldnames=fieldnames)

            # writerow() will write a row in your csv file
          #  writer.writerow({'ampHrs': ampHrs,'apps': apps,'HDSize':HDSize, 'wireless':wireless, 'GHzs': GHzs, 'GBs': GBs,'retailPrice':retailPrice,'Ins':Ins})
        # And you return a text or a template, but if you don't return anything
        # this code will never work.
        return render_template('dash-info.html')
    else:
        return render_template('error-page.html')
    


#Dashboard Result Page
@app.route('/result', methods=['POST','GET'])
def dash_result():
    if request.method == 'POST':
        EcoRange = request.form['EcoRange']
        LowRange = request.form['LowRange']
        DecRange = request.form['DecRange']
        SaveRange = request.form['SaveRange']
        BenRange = request.form['BenRange']
        UseRange = request.form['UseRange']
        IdeaRange = request.form['IdeaRange']
        ClientRange = request.form['ClientRange']
        NeedRange = request.form['NeedRange']
        PotRange = request.form['PotRange']
        MechRange = request.form['MechRange']
        ResRange = request.form['ResRange']
        novel = request.form.get('novel')
        fieldnames = ['EcoRange','LowRange','DecRange','SaveRange','BenRange','UseRange','IdeaRange','ClientRange','NeedRange','PotRange','MechRange','ResRange','novel']
        with open('newtech.csv', 'w', newline="") as inFile:
            #writer = csv.DictWriter(inFile, fieldname=["Date", "temperature 1", "Temperature 2"])
            #writer.writeheader()
            writer = csv.DictWriter(inFile, fieldnames=fieldnames)
            writer.writerow({'EcoRange':EcoRange,'LowRange':LowRange,'DecRange':DecRange,'SaveRange':SaveRange,'BenRange':BenRange,'UseRange':UseRange,'IdeaRange':IdeaRange,'ClientRange':ClientRange,'NeedRange':NeedRange,'PotRange':PotRange,'MechRange':MechRange,'ResRange':ResRange,'novel':novel})
            inFile.close()
            # Importing the libraries
            

            # Importing the dataset
            dataset = pd.read_csv('Laptopsdata.csv')
            data = pd.read_csv('Newdata.csv')
            hg = np.array(data)
            dataset.loc[576, 'Battery_Life':'Sales'] = hg.tolist()[0]
            # dataset = pd.concat([dataset, data], axis=0,  ignore_index=True)
            dataset.dropna(inplace=True)
            X = dataset.iloc[:, 0:8].values
            # find the right number of clusters
            wcss = []
            for i in range(1, 10):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            # plt.plot(range(1, 20), wcss)
            # plt.title('The Elbow Method')
            # plt.xlabel('Number of clusters')
            # plt.ylabel('WCSS')
            # fig1 = plt.gcf()
            # plt.show()

            # fig1.savefig("static/graph1.png")
            # Fitting K-Means to the dataset
            kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
            y_kmeans = kmeans.fit_predict(X)
            dataset['Cluster'] = y_kmeans
            # finding cluster number of the new data
            l = dataset.iloc[575, 9]
            # Grouping by cluster Number
            g = dataset.groupby('Cluster')
            df = g.get_group(l)
            # a = df.iloc[:, :-2].values
            # b = df.iloc[:, 8].values
            # train test split
            a_train = df.iloc[:-1, :-2].values
            b_train = df.iloc[:-1, 8].values
            a_test = df.iloc[-1:, :-2].values
            b_test = df.iloc[-1:, 8].values
            # model fitting
            # from sklearn.cross_validation import train_test_split
            # a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.1, random_state = 0)
            regressor= RandomForestRegressor(n_estimators=400,random_state=1)
            regressor.fit(a_train, b_train)
            # Predicting the Test set results
            b_pred = regressor.predict(a_test)
            b_pred = np.round(b_pred)
            print(b_pred)
            y_new_pred = b_pred
            # from sklearn.metrics import mean_squared_error
            # from math import sqrt
            # ms = mean_squared_error(b_test, b_pred)
            # ms1 = sqrt(mean_squared_error(b_test, b_pred ))

            newtech = pd.read_csv('newtech.csv', header=None)
            newtech = np.array(newtech)
            aa1 = newtech[:, 0]
            aa2 = newtech[:, 1]
            aa3 = newtech[:, 2]
            aa4 = newtech[:, 3]
            aa5 = newtech[:, 4]
            aa6 = newtech[:, 5]
            aa7 = newtech[:, 6]
            aa8 = newtech[:, 7]
            aa9 = newtech[:, 8]
            aa10 = newtech[:, 9]
            aa11 = newtech[:, 10]
            aa12 = newtech[:, 11]
            degree = newtech[:, 12]
            relad = degree * 5 * (aa1 + aa2 + aa3 + aa4 + aa5)
            compat = degree * 4 * (aa6 + aa7 + aa8)
            complexi = degree * 6 * (aa9)
            trial = degree * 3 * aa10
            obser = degree * 4 * aa11
            maxr = 9 * 5 * 45
            maxcp = 9 * 4 * 21
            maxc = 9 * 6 * 7
            maxt = 9 * 3 * 7
            maxo = 9 * 4 * 7
            max1 = maxr + maxcp + maxc + maxt + maxo
            min1 = 5 + 4 + 4 + 3 + 4
            '''normr=(((relad-minr)/(maxr-minr))*(1.25-0.75))+0.75
            normcp=(((compat-mincp)/(maxcp-mincp))*(1.25-0.75))+0.75
            normc=(((complexi-minc)/(maxc-minc))*(1.25-0.75))+0.75
            normt=(((trial-mint)/(maxt-mint))*(1.25-0.75))+0.75
            normo=(((maxo-mino)/(maxo-mino))*(1.25-0.75))+0.75
            '''
            ip = relad + compat + complexi + trial + obser
            ip2 = (((ip - min1) / (max1 - min1)) * (1.25 - 0.75)) + 0.75

            # print(min1)
            # print(max1)
            print(ip)
            print(ip2)
            newdemand = int(y_new_pred * ip2)
            newdemand=math.ceil(newdemand)
            print("y_new_pred:",y_new_pred)
            print("newdemand:",newdemand)

            import matplotlib.pyplot as plt
            import pylab as pl

            diff = newdemand - y_new_pred
            nd1 = newdemand / 0.025
            x = [1, 2, 3, 4, 5]
            dds = np.array([nd1 * 0.025, nd1 * 0.135, nd1 * 0.34, nd1 * 0.34, nd1 * 0.16])

            pl.plot(x, dds, "-o", label = "1- early adopters \ 2- jkdbsvjkn")
            for x, y in zip(x, dds):
                pl.text(x, y, str(x), color="black")
            pl.margins(0.1)

            plt.ylabel('SALES')
            plt.xlabel('1-Innovators 2-Early Adopter 3-Early Majority 4-Late Majority 5-Laggards')

            plt.title('Diffusion Curve')


            fig2 = plt.gcf()
            plt.show()
            fig2.savefig("static/graph2.png")
            dds = dds.astype(int)
            dds2 = dds

            diffusionData = {
                "x": x,
                "dds": dds.tolist(),
                "labels": ["1-Innovators","2-Early Adopters","3-Early Majority","4-Late Majority","5-Laggards"]
            }
            # plt.savefig('plot')
            # plt.clf()
            # print(dds2)
            # plt.xlabel()
            # from scipy.interpolate import spline

            # xnew = np.linspace(dds2.min(),dds2.max(),300) #300 represents number of points to make between T.min and T.max

            # power_smooth = spline(dds2,xnew,1)

            # plt.plot(xnew,power_smooth)
            # plt.show()
            # plt.title('Spiline Interpolation')
            # plt.xlabel('X-axis')
            # plt.ylabel('Y-axis')
            # fig4 = plt.gcf()
            # plt.show()

            # fig4.savefig("static/graph4.png")
            

            import matplotlib.pyplot as plt;
            plt.rcdefaults()
            objects = ('Sales based on common features', 'After New features')
            y_pos = np.arange(len(objects))
            y_new_pred=y_new_pred.astype(int)
            print("y_new_pred:",y_new_pred)
            print("newdemand:",newdemand)

            performance = [y_new_pred, newdemand]
            performance_array = np.array(performance)
            performance_list = performance_array.tolist()
            performance_int = [int(x[0]) if isinstance(x, np.ndarray) else x for x in performance_list]
            performance_int=np.nan_to_num(performance_int).tolist()
            print("performance_list",performance_int)


            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('sales in number of units')
            plt.title('Demand')
            fig3 = plt.gcf()
            plt.show()
            fig3.savefig("static/graph3.png")
            y_pos=np.nan_to_num(y_pos).tolist()
            demandData = {
                "y_pos": y_pos,
                "performance": performance_int,
                "labelData": ["Common features", "New features"]
            }


           


            return render_template('dash-result.html', diffusionData=diffusionData,demandData=demandData)
        
    else:
        return render_template('error-page.html')


if __name__ == '__main__':
    app.run()


