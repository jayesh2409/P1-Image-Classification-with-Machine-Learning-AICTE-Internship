# P1-Image-Classification-with-Machine-Learning-AICTE-Internship
AICTE Internship Project
# I have created this project model to classify the images so that the actual object in the image will get classified and the model will also predict its actual accuracy of prediction of that object. 

# Here the objects can be cat, dog, truck, bird.

# Two models namely CNN model and MobileNetV2(image net) are used here

# CNN model predects the image but with less accuracy and has slower prediction.
# Meanwhile the MobileNetV2(image net) model is much faster as compared to CNN model as it also provides the class of that particular object and with accurate to 90% accuracy.

# Here i ave created an app using the Streamlit module.

# I made this app using the Google Colab by using the module pyngrok, because it helped to directly run this streamlit app output on the browser. 
#As the google colab does not provide direct output for the streamlit application. Therefor the pyngrok helped me to create a tunnel from google colab and helped me to get the output

# following are the steps to run the app on the google colab directly by streamlit and pyngrok.
#STEP 1:	(importing all files of the project)
	
	>>> from google.colab import files as fs
	>>> uploaded=fs.upload()

#STEP 2:	(install streamlit every time)

	>>> !pip install streamlit

#STEP 3:	(install ngrok every time)

	>>> !pip install pyngrok

#STEP 4: (add the ngork api key compulsory)

	>>> !ngrok authtoken your_key_here

#STEP 5: (run the app)

	>>> !streamlit run app_name.py

#STEP 6: (create tunnel to run the streamlit on browser)

	>>> from pyngrok import ngrok
	>>> public_url = ngrok.connect(8501)
	>>> print(f"Streamlit app is live at: {public_url}")

#STEP 7: (get output)

	just click on the link which gets generated in the output of STEP 6
