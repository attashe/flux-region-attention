import io
import json
import flask

import time
import yaml
import uuid
import traceback
import numpy as np

from pathlib import Path
from collections import defaultdict
from flask import Flask, send_file
from flask_socketio import SocketIO, send, emit, join_room
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

from threading import Thread
from multiprocessing import Process, Lock, Queue
from flux.generate import FluxInference

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')


class Context:
    debug: bool = False
    inference_process: Process = None
    queue_in: Queue = None
    queue_out: Queue = None
    output_folder: str = 'output'
    output_path: Path = Path('output')
    lock = Lock()
    image_storage = {}
    users_waiting_flag = defaultdict(lambda: False)

def generate_image_wrapper(t2i_model: FluxInference, payload: dict):
    bboxes = payload['prompt']['bboxes']
    masks = []
    subprompts = []
    
    for bbox in bboxes:
        mask = Image.new('L', (payload['prompt']['width'], payload['prompt']['height']), 0)
        mask_arr = np.array(mask)
        
        # Draw the bounding box
        mask_arr[bbox['y']:bbox['y']+bbox['height'], bbox['x']:bbox['x']+bbox['width']] = 255
        mask = Image.fromarray(mask_arr)
        
        # Debug save the mask
        # mask.save(f'mask_{bbox["idx"]}.png')
        
        masks.append(mask)
        subprompts.append(bbox['caption'])
    
    image = t2i_model.inference_bbox(
        prompt=payload['prompt']['positive'], negative_prompt=payload['prompt']['negative'],
        masks=masks, subprompts=subprompts,
        aspect_ratio='1:1', seed=int(payload['prompt']['seed']), steps=int(payload['prompt']['steps']),
        guidance=float(payload['prompt']['cfg']),
        height=payload['prompt']['height'], width=payload['prompt']['width']
    )
    
    return Image.fromarray(image)

@app.route("/")
def index():
    return flask.render_template('base.html')

@socketio.on('generate_image')
def generate_image_socket(payload):
    print(f'Payload: {payload}')
    
    user_id = payload['user_id']
    if Context.users_waiting_flag[user_id]:
        return 'User is already waiting for an image', 400, {'ContentType':'text/plain'}
    Context.users_waiting_flag[user_id] = True
    
    image_id = str(uuid.uuid4())
    join_room(image_id)
    
    Context.queue_in.put((user_id, image_id, payload))
    print(f'Image ID: {image_id}')
    
    response = json.dumps({'image_id': image_id})
    
    # Start the queue listener thread
    listener = Thread(target=queue_listener, args=(Context.queue_out,))
    listener.daemon = True
    listener.start()
    
    emit('image_generated', response, room=image_id)

@app.route('/get_image/<input_image_id>', methods=['GET'])
def get_image(input_image_id):
    # Get the image
    image = Context.image_storage.get(input_image_id)
    if image is None:
        return 'Image not found', 404, {'ContentType':'text/plain'}
    
    # Save the image to a bytes buffer
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Return the image
    return send_file(img_buffer, mimetype='image/png')

def inference_process(inference_config: dict, queue_in: Queue, queue_out: Queue):
    # Load the model
    t2i_model = FluxInference(inference_config)
    # print('Mocking the model')
    
    # Wait for the first payload
    while True:
        user_id, image_id, payload = queue_in.get()
        
        # Check if the payload is a stop signal
        if payload is None:
            break
        
        try:
            image = generate_image_wrapper(t2i_model, payload)
        except Exception as e:
            print(f'Error generating image: {e}')
            traceback.print_exc()
            image = Image.new('RGB', (256, 256), (255, 0, 0))
        finally:
            # Put the image ID in the output queue
            queue_out.put((user_id, image_id, image))
            
        # image = Image.new('RGB', (256, 256), (255, 0, 0))
        # time.sleep(5)
        
        save_log_image(image_id, image, payload)


def save_log_image(ids, image, payload):
    # Save the image with the image ID and all payload as metadata
    metadata = PngInfo()
    metadata.add_text('payload', json.dumps(payload))
    image.save(Context.output_path / f'image_{ids}.png', pnginfo=metadata)
    
    # Save payload to a file
    with open(Context.output_path / f'payload_{ids}.json', 'w') as f:
        json.dump(payload, f)
  
def queue_listener(queue_out: Queue):
    user_id, image_id, image = queue_out.get()
    
    Context.image_storage[image_id] = image
    Context.users_waiting_flag[user_id] = False
    
    socketio.emit('update_image', {'image_id': image_id}, namespace='/', room=image_id)

def main(args):
    # Load the config
    with open(args.config, 'r') as f:
        inference_config = yaml.safe_load(f)
    
    Context.queue_in = Queue()
    Context.queue_out = Queue()
    
    # Start the new process
    predictor_daemon = Process(target=inference_process,
                               args=(inference_config, Context.queue_in, Context.queue_out))
    predictor_daemon.daemon = True
    predictor_daemon.start()
    
    Path(Context.output_folder).mkdir(exist_ok=True)
    
    # Serve flask app
    socketio.run(app, host='127.0.0.1',
                 port=args.port, debug=False, log_output=args.debug)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='flux_dev_config.yaml')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    main(args)