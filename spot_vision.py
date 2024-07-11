# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image capture tutorial."""

import argparse
import sys
import cv2
import numpy as np
from scipy import ndimage
import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
import boto3
from botocore.exceptions import ClientError
import base64
import json
import webbrowser
import os
import pygame
import time
import io
import sounddevice
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import asyncio
from flask import Flask, render_template_string, request
import jsonify

# Defining Spot camera rotation angles
ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


# Formatting strings
def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]


# More string formatting
def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


# Setup Bedrock client and Claude model
client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"


# Defining image output names
front_l = 'frontright_fisheye_image.jpg'
front_r = 'frontleft_fisheye_image.jpg'
back =   'back_fisheye_image.jpg'
left = 'left_fisheye_image.jpg'
right = 'right_fisheye_image.jpg'


# Get image data from image
def get_img_data(path):
    with open(path, "rb") as image_file:
        content = image_file.read()
    image1_data = base64.b64encode(content).decode("utf-8")
    return(image1_data)


# Synthesize speech with Amazon Polly 
def synthesize_speech(text, output_filename, voice_id, language_code):
    # Create a Polly client
    polly_client = boto3.client('polly')

    # Request speech synthesis
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId=voice_id,
        LanguageCode=language_code,
        Engine='neural'
    )
    
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Read the audio stream and play it
    audio_stream = io.BytesIO(response['AudioStream'].read())
    pygame.mixer.music.load(audio_stream, 'mp3')
    pygame.mixer.music.play()
    
    # Keep the script running until the playback is finished
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


# Get description of images with Claude
def get_description():
    # Setup Image Prompt
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Front Left Camera:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_img_data(front_l),
                    },
                },
                {
                    "type": "text",
                    "text": "Front Right Camera:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_img_data(front_r),
                    },
                },
                {
                    "type": "text",
                    "text": "Left Camera:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_img_data(left),
                    },
                },
                {
                    "type": "text",
                    "text": "Right Camera:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_img_data(right),
                    },
                },
                {
                    "type": "text",
                    "text": "Back Camera:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_img_data(back),
                    },
                },
                {
                    "type": "text",
                    "text": "You are a Boston Dynamics SPOT Robot. You have five fisheye cameras positioned at your front left, front right, left, right, and back sides. You are given each of these images. From these images, describe what you see. Point out any specific features or people that you see. List as many details about the space as possible. Do not describe what each individual camera sees, but instead imagine the images were composited together into a 360 degree view and base what you see on that. Do not mention anything about cameras or compositing, simply state what you see. Start your reply with 'I see'. You are programmed with humor. Make sure that your response is funny and contains light-hearted jokes about what you see. Do not be mean or insult anyone or anything. Instead, be positive, upbeat, and kind with your humour. Do not mention mad scientists, supervillains. Ensure your response is less than three sentences in length."
                }
            ],
        }
    ]

    # Setup Request 
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 5120,
        "temperature": 0.5,
        "messages": messages
    }

    request = json.dumps(native_request)
    
    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())   

    # Extract and print the response text.
    response_text = model_response["content"][0]["text"]
    print(response_text)
    return(response_text)


# Create HTML page to view Spot output
def create_and_open_html(content: str, filename: str = 'output.html'):
    # Define the HTML structure
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Display Content</title>
    </head>
    <body>
        <h1>{content}</h1>
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open(filename, 'w') as file:
        file.write(html_content)

    # Get the absolute path of the file
    file_path = os.path.abspath(filename)

    # Open the file in the default web browser
    webbrowser.open(f'file://{file_path}')


def capture_images():
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--list', help='list image sources', action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    parser.add_argument('--image-sources', help='Get image from source(s)', action='append')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument(
        '--pixel-format', choices=pixel_format_type_strings(),
        help='Requested pixel format of image. If supplied, will be used for all sources.')

    options = parser.parse_args()

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(options.image_service)

    # Raise exception if no actionable argument provided
    if not options.list and not options.image_sources:
        parser.error('Must provide actionable argument (list or image-sources).')

    # Optionally list image sources on robot.
    if options.list:
        image_sources = image_client.list_image_sources()
        print('Image sources:')
        for source in image_sources:
            print('\t' + source.name)

    # Optionally capture one or more images.
    if options.image_sources:
        # Capture and save images to disk
        pixel_format = pixel_format_string_to_enum(options.pixel_format)
        image_request = [
            build_image_request(source, pixel_format=pixel_format)
            for source in options.image_sources
        ]
        image_responses = image_client.get_image(image_request)

        for image in image_responses:
            num_bytes = 1  # Assume a default of 1 byte encodings.
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
                extension = '.png'
            else:
                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                    num_bytes = 3
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                    num_bytes = 4
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    num_bytes = 1
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                    num_bytes = 2
                dtype = np.uint8
                extension = '.jpg'

            img = np.frombuffer(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                try:
                    # Attempt to reshape array into an RGB rows X cols shape.
                    img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
                except ValueError:
                    # Unable to reshape the image data, trying a regular decode.
                    img = cv2.imdecode(img, -1)
            else:
                img = cv2.imdecode(img, -1)

            if options.auto_rotate:
                img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

            # Save the image from the GetImage request to the current directory with the filename
            # matching that of the image source.
            image_saved_path = image.source.name
            image_saved_path = image_saved_path.replace(
                '/', '')  # Remove any slashes from the filename the image is saved at locally.
            cv2.imwrite('static/' + image_saved_path + extension, img)


class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                transcript = alt.transcript
                print(transcript)
                # Check if the specific phrase is in the transcript
                if "looking at?" in transcript:
                    # main()
                    # webbrowser.open('127.0.0.1:5000/spot', new=2)
                    print("TEST OUTPUT")


# Stream microphone input
async def mic_stream():
    # This function wraps the raw input stream from the microphone forwarding
    # the blocks to an asyncio.Queue.
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

    # Be sure to use the correct parameters for the audio stream that matches
    # the audio formats described for the source language you'll be using:
    # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )
    # Initiate the audio stream and asynchronously yield the audio chunks
    # as they become available.
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status


# Write audio chunks
async def write_chunks(stream):
    # This connects the raw audio chunks generator coming from the microphone
    # and passes them along to the transcription stream.
    async for chunk, status in mic_stream():
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()


# Transcribe speech
async def basic_transcribe():
    # Setup up our client with our chosen AWS region
    client = TranscribeStreamingClient(region="us-west-2")

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )

    # Instantiate our handler and start processing events
    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(stream), handler.handle_events())


def main():
    # Capture Spot images
    capture_images()

    # Get description of the images
    ai_response = get_description()

    # View response as HTML page
    # create_and_open_html(ai_response)

    # Synthesize and play speech of response
    synthesize_speech(ai_response, output_filename='speech.mp3', voice_id='Matthew', language_code='en-US')

    return True


app = Flask(__name__)

@app.route('/')
def index():
    # HTML template with a button that triggers the trigger() function
    html_template = '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>What Spot Sees</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
          }
          .container {
            text-align: center;
            margin-top: 50px;
            max-width: 90%;
          }
          button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 20px 0;
            cursor: pointer;
            border-radius: 4px;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>What is Spot Looking At?</h1>
          <form action="/spot" method="post">
            <button type="submit">Activate Spot's Vision</button>
          </form>
        </div>
      </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/spot', methods=['GET', 'POST'])
def spot():
    # Function to be triggered by the button press
    print("Button was pressed!")
    # capture_images()
    ai_response = get_description()
    ai_response = ai_response.replace("'", "").replace('"', "").replace('\n', ' ').replace('\r', ' ')
    # ai_response = '''*beep boop* I see a rather dreary and drab environment. The floor appears to be made of some kind of industrial-grade linoleum that looks like it's seen better days. The walls are a depressing shade of gray, likely chosen to match the soul-crushing monotony of whatever activities take place here. Ah, and what's this? A mysterious cardboard box, likely containing the dreams and aspirations of some poor soul who's been relegated to this dismal space. I bet it's filled with nothing but broken dreams and the faint scent of despair. And let's not forget the piece de resistance - a strange black-and-white target on the wall. Is it for target practice? A subtle reminder that this place is a soul-crushing void where dreams come to die? The possibilities are endless, and all of them equally bleak. Overall, I'd give this place a solid 2 out of 10 on the places I'd want to spend my time scale. But hey, at least it's not a windowless van, right? *beep boop* Initiating sarcasm subroutine... complete.'''

    # HTML template for the trigger page with a button to go back to the main page
    html_template = f'''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>What Spot Sees</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
          }}
          .container {{
            text-align: center;
            margin-top: 50px;
            max-width: 90%;
          }}
          .image-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 20px 0;
          }}
          .image-grid img, .centered-row img {{
            width: 100%;
            height: auto;
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
          }}
          .centered-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            justify-content: center;
          }}
          button {{
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 20px 0;
            cursor: pointer;
            border-radius: 4px;
          }}
        </style>
        <script>
          document.addEventListener("DOMContentLoaded", function() {{
            console.log("Page loaded. Sending AJAX request.");
            fetch('/synthesize_speech', {{
              method: 'POST',
              headers: {{
                'Content-Type': 'application/json'
              }},
              body: JSON.stringify({{ ai_response: "{ai_response}" }})
            }}).then(response => response.json()).then(data => {{
              console.log(data.message);
            }}).catch(error => console.error("Error:", error));
          }});
        </script>
      </head>
      <body>
        <div class="container">
          <h1>What Spot Sees</h1>
          <p>{ai_response}</p>
          <div class="image-grid">
            <img src="{{{{url_for('static', filename='frontright_fisheye_image.jpg')}}}}" alt="Front Right Image">
            <img src="{{{{url_for('static', filename='frontleft_fisheye_image.jpg')}}}}" alt="Front Left Image">
          </div>
          <div class="centered-row">
            <img src="{{{{url_for('static', filename='left_fisheye_image.jpg')}}}}" alt="Left Image">
            <img src="{{{{url_for('static', filename='back_fisheye_image.jpg')}}}}" alt="Back Image">
            <img src="{{{{url_for('static', filename='right_fisheye_image.jpg')}}}}" alt="Right Image">
          </div>
          <form action="/" method="get">
            <button type="submit">Go back to main page</button>
          </form>
        </div>
      </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/synthesize_speech', methods=['POST'])
def synthesize_speech_endpoint():
    try:
        data = request.get_json()
        if not data or 'ai_response' not in data:
            print("Invalid data received")
            return jsonify({'message': 'Invalid data'}), 400
        ai_response = data['ai_response']
        print(f"Synthesizing speech for response: {ai_response}")
        synthesize_speech(ai_response, output_filename='speech.mp3', voice_id='Matthew', language_code='en-US')
        return jsonify({'message': 'Speech synthesis complete'}), 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
    

# loop = asyncio.get_event_loop()
# loop.run_until_complete(basic_transcribe())
# loop.close()


