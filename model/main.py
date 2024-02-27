import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle as pickle

def create_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  
  model = LogisticRegression()
  model.fit(X_train, y_train)
  
  y_pred = model.predict(X_test)
  y_proba = model.predict_proba(X_test)
  confidence = np.max(y_proba, axis=1)  # Valor de confianza
  
  print('Precision del modelo: ', accuracy_score(y_test, y_pred))
  print("Reporte de Clasificacion: \n", classification_report(y_test, y_pred))
  print("Valores de confianza: \n", confidence)
  
  return model, scaler

def create_decision_tree_model(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    confidence = np.max(y_proba, axis=1)  # Valor de confianza
    
    print('Precision del arbol: ', accuracy_score(y_test, y_pred))
    print("Reporte de Clasificacion: \n", classification_report(y_test, y_pred))
    print("Valores de confianza: \n", confidence)
    
    return model

def apply_kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)

    return kmeans, labels


def get_clean_data():
  data = pd.read_csv("data/data.csv")
  
  data = data.drop([ 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

def main():
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    logistic_model, scaler = create_model(data)
    decision_tree_model = create_decision_tree_model(X_train, y_train, X_test, y_test)
    
    kmeans_model, labels = apply_kmeans(X_scaled)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(logistic_model, f)
        
    with open('model/tree_model.pkl', 'wb') as f:
        pickle.dump(decision_tree_model, f)
        
    with open('model/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
  

if __name__ == '__main__':
  main()