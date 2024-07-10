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

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)

# Setup Bedrock client and Claud model
client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# System prompt
prompt = "Describe the image."

front_l = 'frontright_fisheye_image.jpg'
front_r = 'frontleft_fisheye_image.jpg'
back =   'back_fisheye_image.jpg'
left = 'left_fisheye_image.jpg'
right = 'right_fisheye_image.jpg'


# Get image data
def get_img_data(path):
    with open(path, "rb") as image_file:
        content = image_file.read()
    image1_data = base64.b64encode(content).decode("utf-8")
    return(image1_data)

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
                    "text": "You are a Boston Dynamics SPOT Robot. You have five fisheye cameras positioned at your front left, front right, left, right, and back sides. You are given each of these images. From these images, describe what you see. Point out any specific features or people that you see. List as many details about the space as possible. Do not describe what each individual camera sees, but instead imagine the images were composited together into a 360 degree view and base what you see on that. Do not mention anything about cameras or compositing, simply state what you see. Start your reply with 'I see'"
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

def main():
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
            cv2.imwrite(image_saved_path + extension, img)
    ai_response = get_description()
    # create_and_open_html(ai_response)
    synthesize_speech(ai_response, output_filename='speech.mp3', voice_id='Matthew', language_code='en-US')
    return True

sample_text = """
I see a large, open space that appears to be a lobby or entrance area. The floor is made of a smooth, gray material, likely a type of terrazzo or concrete. There are several chairs and stools arranged along the walls, suggesting this is a waiting or seating area.

On the right side of the space, there is a counter or desk with a display screen and some equipment, possibly for check-in or security purposes. The walls are made of a light-colored wood paneling, creating a warm and modern aesthetic.

The space is well-lit, with overhead lighting fixtures and natural light streaming in through the windows on the back wall. The windows offer a view of a city skyline, indicating this is likely located in an urban setting.

Overall, the space appears to be clean, organized, and designed with a focus on functionality and minimalism. There are no obvious signs of human presence, aside from the equipment and furnishings, suggesting this is an unoccupied area at the moment."""

# synthesize_speech(sample_text, output_filename='speech.mp3', voice_id='Gregory', language_code='en-US')
if __name__ == '__main__':
    if not main():
        sys.exit(1)