#correct oreder in dataframe
import pickle

def prediction_model(pclass,sex, age, sibsp, parch, fare, embarked, title):
    x=[[pclass,sex, age, sibsp, parch, fare, embarked, title]]
    randomforest=pickle.load(open('titanic_model.sav','rb'))
    prediction=randomforest.predict(x)
    if prediction == 0:
        prediction=' Unfortunately the person did Not Survived'
    elif prediction == 1:
        prediction=' The person Survived'
    else:
        prediction='Error'
    return prediction
