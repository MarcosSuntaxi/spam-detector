import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Leer el dataset
df = pd.read_csv("dataset.csv")

# Vectorizar los mensajes
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['mensaje'])
y = df['etiqueta']

# Dividir entre entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
print("Precisión del modelo:", accuracy_score(y_test, y_pred))

# Probar con un nuevo mensaje
nuevo = ["¡Método para ganar la lotería!"]
nuevo_vector = vectorizer.transform(nuevo)
prediccion = model.predict(nuevo_vector)
print("¿Es spam?:", prediccion[0])
