# text-image-sound

This is application gets a prompt from the users and generates an image using stable-diffusion. 

After the image is generated, the it uses the pixel information to generate a short music using hilbert-space filling curves.

The application has 3 main components:

 1- Python backend using Flask

 2- MongoDB as database to store the image and sound files temporarily 
 
 3- Nginx server to serve the simple frontend

To run everything, you can use docker-compose

        cd src

        docker-compose up --build 

and go to the localhost:80 to access the ui. (generating image may take some time)

![alt text](examples/image.png)

["summer"s sound](examples/summer.wav)

### functions in the backend: 

        generate-image : generates an image based on the user prompt and returns the objectID and the image data

        generate-sound : generates a sound file based on the generaetd image's objectID. 

        save-image : saves the generated image to the database, otherwise the generated images and sounds will be deleted after 5 minutes.

        get-images : returns the images and sounds that were saved in the database

