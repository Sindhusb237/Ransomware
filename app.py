import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #random-forest
from sklearn.naive_bayes import GaussianNB #naive-bayes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

global data,preprocessed_data
data=pd.read_csv("./ProjectData/ugransom.csv")
preprocessed_data=data.copy(True)
objList = data.select_dtypes(include = "object").columns

le = LabelEncoder()

for feat in objList:
    preprocessed_data[feat] = le.fit_transform(preprocessed_data[feat].astype(str))

X=preprocessed_data.drop(['Prediction'],axis=1)
y=preprocessed_data['Prediction']

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def eda():
    st.markdown("<h2><u>Data Summary:</u></h2>",True)
    st.dataframe(data.describe().T)

    fig,ax=plt.subplots()
    sns.countplot(x=data['IPaddress'],data=data,ax=ax)
    ax.set_title(label="Bar Graph of IPaddress")
    st.pyplot(fig)

    fig,ax=plt.subplots()
    sns.countplot(x=data['Prediction'],data=data,ax=ax)
    ax.set_title(label="Bar Graph of Prediction")
    st.pyplot(fig)

    fig,ax=plt.subplots()
    sns.countplot(x=data['Protocol'],data=data,ax=ax)
    ax.set_title(label="Bar Graph of Protocol")
    st.pyplot(fig)

    fig,ax=plt.subplots()
    fig.set_size_inches((15,6))
    sns.countplot(x=data['Family'],data=data,ax=ax)
    ax.set_title(label="Bar Graph of Family")
    ax.set_xticks(ticks=ax.get_xticks(),labels=ax.get_xticklabels(),rotation=45)
    st.pyplot(fig)

    st.markdown("<h2><u>Pre-Processed Data:</u></h2>",True)
    st.dataframe(preprocessed_data[0:10])
    

def logisticRegression():
    #st.write(data.shape)
    st.write("Logistic Regression")
    try:
        logreg=LogisticRegression(solver='lbfgs',max_iter=1000)
        logreg.fit(X_train,y_train)
        logreg_pred=logreg.predict(X_test)

        logreg_accuracy=accuracy_score(logreg_pred,y_test)
        logreg_report=classification_report(logreg_pred,y_test,output_dict=True)
        logreg_matrix = confusion_matrix(logreg_pred, y_test)

        st.write("Model Accuracy: " + str(round(100.0 * logreg_accuracy, 2)) + " %")
        st.dataframe(pd.DataFrame(logreg_report).transpose(), width=800)

        num_wrong = (y_test != logreg_pred).sum()
        st.write('Number of incorrect predictions by Logistic Regression:', num_wrong)

        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(logreg_matrix, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=["0:A", "1:S", "2:SS"], yticklabels=["0:A", "1:S", "2:SS"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix Heatmap of Logistic Regression")
        st.pyplot(plt)
    except Exception as e:
        st.write(f"An error occured: {e}")



def naiveBayes():
    st.write("Naive Bayes")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    
    nb_accuracy = accuracy_score(nb_pred, y_test)
    nb_report = classification_report(nb_pred, y_test,output_dict=True)
    nb_matrix = confusion_matrix(nb_pred, y_test)

    st.write("Model Accuracy: "+str(round(100.0*nb_accuracy,2))+" %")
    st.dataframe(pd.DataFrame(nb_report).transpose(),width=800)

    num_wrong = (y_test != nb_pred).sum()
    st.write('Number of incorrect predictions by Naive Bayes:', num_wrong)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  
    sns.heatmap(nb_matrix, annot=True, fmt="d", cmap="inferno", cbar=False,
                xticklabels=["0:A", "1:S", "2:SS"], yticklabels=["0:A", "1:S", "2:SS"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Heatmap of Naive Bayes")
    st.pyplot(plt)
   

def randomForest():
    st.write("Random Forest")
    rfc=RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    
    rfc_accuracy = accuracy_score(rfc_pred, y_test)
    rfc_report = classification_report(rfc_pred, y_test,output_dict=True)
    rfc_matrix = confusion_matrix(rfc_pred, y_test)

    st.write("Model Accuracy: "+str(round(100.0*rfc_accuracy,2))+" %")
    st.dataframe(rfc_report,width=800)

    num_wrong = (y_test != rfc_pred).sum()
    st.write('Number of incorrect predictions by Naive Bayes:', num_wrong)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  
    sns.heatmap(rfc_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["0:A", "1:S", "2:SS"], yticklabels=["0:A", "1:S", "2:SS"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Heatmap of RandomForestClassifier")
    st.pyplot(plt)

def adaBoost():
    st.write("Ada Boost")
    ada=AdaBoostClassifier(n_estimators=100,random_state=0)
    ada.fit(X,y)
    ada_pred=ada.predict(X_test)

    ada_accuracy = accuracy_score(ada_pred, y_test)
    ada_report = classification_report(ada_pred, y_test,output_dict=True)
    ada_matrix = confusion_matrix(ada_pred, y_test)

    st.write("Model Accuracy: "+str(round(100.0*ada_accuracy,2))+" %")
    st.dataframe(ada_report,width=800)

    num_wrong = (y_test != ada_pred).sum()
    st.write('Number of incorrect predictions by Naive Bayes:', num_wrong)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  
    sns.heatmap(ada_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["0:A", "1:S", "2:SS"], yticklabels=["0:A", "1:S", "2:SS"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Heatmap of RandomForestClassifier")
    st.pyplot(plt)



st.sidebar.title("Menu")
page=st.sidebar.radio(" ",["Home","Explorative Data Analysis","Logistic Regression","Naive Bayes","Random Forest","Ada Boost","About"])

st.title("AI-Defence: Proactive Ransomware Detection via Machine Learning")

if page=="Home":
    st.markdown("<h2><u>Overview:</u></h2>",True)
    st.markdown("Our Project Aims to compare the performance of various machine learning models in predicting the type of ransomware.\
    The dataset we are using for this project is the UGRansome dataset. The dataset consists of "+str(data.shape[0])+" rows and \
    "+str(data.shape[1])+" columns. Each Column represents a unique feature corresponding to a ransomware application. We have employed \
    four Machine Learning Models for a comparitive analysis in their performance:<br>\
    1. Logistic Regression<br>\
    2. Naive Bayes<br>\
    3. Random Forest<br>\
    4. Ada Boost<br>\
    ",True)

    st.markdown("<h2><u>Dataset:</u></h2>",True)
    st.markdown("<b>Name:</b> UGRansome",True)
    st.markdown("<b>Data Format:</b> CSV",True)
    st.markdown("<b>Number of Rows:</b> "+str(data.shape[0]),True)
    st.markdown("<b>Number of Columns:</b> "+str(data.shape[1]),True)
    st.markdown("<b>Meaning of various Feature Columns:</b>",True)
    st.write("""
1) Time: Quantitative column with integers indicating the timestamp of network attacks.

2) Protocol: Qualitative/categorical column representing the network protocol used (e.g., TCP, UDP).

3) Flag: Qualitative/categorical column indicating network connection status (e.g., SYN, ACK).

4) Family: Qualitative/categorical column describing network intrusion category.

5) Clusters: Quantitative column with integers denoting event clusters or groups.

6) SeedAddress: Qualitative/categorical column representing formatted ransomware attack links.

7) ExpAddress: Qualitative/categorical column indicating original ransomware attack links.

8) BTC: Numeric column with values related to Bitcoin transactions in attacks.

9) USD: Numeric column indicating financial damages in USD caused by attacks.

10) Netflow Bytes: Quantitative column with integers showing bytes transferred in network flow.

11) IPaddress: Qualitative column with IP addresses associated with network events.

12) Threats: Qualitative column representing the nature of threats or intrusions.

13) Port: Quantitative column indicating network port number in events.

14) Prediction: This is the target variable (Qualitative/categorical column indicating predictive model outcomes (anomaly (A), signature (S), and synthetic signature (SS)))
    
    """)

    st.markdown("<h2><u>Sample Data:</u></h2>",True)
    st.write("Below Table Shows the first 10 rows of the dataset ugransom.csv.")
    st.dataframe(data[0:10])


elif page=="About":
    st.markdown("<h2><u>Guide:</u></h2>",True)
    name,designation=st.columns(2)

    name.write("Prof. Sahana M P")

    designation.write("Assistant Professor")

    st.markdown("<h2><u>Contributors:</u></h2>",True)
    name,usn=st.columns(2)
    
    name.write("1. Gowthami R")
    name.write("2. Sandhya T S")
    name.write("3. Sindhu S B")
    

    usn.write("1DS21CS216")
    usn.write("1DS21CS220")
    usn.write("1DS21CS226")
    usn.write("1DS21CS159")

elif page=="Explorative Data Analysis":
    eda()
elif page=="Logistic Regression":
    logisticRegression()
elif page=="Naive Bayes":
    naiveBayes()
elif page=="Random Forest":
    randomForest()
elif page=="Ada Boost":
    adaBoost()

