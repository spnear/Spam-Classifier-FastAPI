# Load the libraries
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel

# Load the model
spam_clf = load(open('./models/spam_detector_model.pkl','rb'))

# Load vectorizer
vectorizer = load(open('./vectors/vectorizer.pickle', 'rb'))

class Text(BaseModel):
    text_to_analyze: str

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}


# Define the route to the sentiment predictor

@app.post("/predict_sentiment")
async def prediction(text: Text):

    if(not(text.text_to_analyze)):
        print("No se ingresó un un texto")
        raise HTTPException(status_code=400, detail = "Please Provide a valid text message")
    
    else:

        print("Texto a analizar:",text.text_to_analyze)

        prediction = spam_clf.predict(vectorizer.transform([text.text_to_analyze]))


        if(prediction[0] == 0):
            polarity = "Ham"
            print("Resultado: No Spam")

        elif(prediction[0] == 1):
            polarity = "Spam"
            print("Resultado: Spam")
            
        return {
                "text_message": text.text_to_analyze, 
                "sentiment_polarity": polarity
            }

