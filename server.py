import flask
import imutil
import gym
import numpy as np

app = flask.Flask(__name__)

class GameEnv(gym.Env):
    def __init__(self, name):
        super().__init__()
        self.env = gym.make(name)
        self.env.unwrapped.frameskip = 1
        self.last_state = self.env.reset()

    def step(self, action):
        new_state, reward, done, info = self.env.step(action)

        # Max the previous two frames to remove flickering
        state = np.maximum(new_state, self.last_state)

        self.last_state = new_state
        return state, reward, done, info

    def reset(self):
        return self.env.reset()



def reset_game():
    global env
    env = GameEnv('CentipedeDeterministic-v4')
    state = env.reset()
    imutil.show(state, filename='static/screenshot.jpg')

@app.route('/')
def route_index():
    return flask.render_template('index.html', num_actions=4)

@app.route('/step', methods=['POST'])
def route_step():
    action = flask.request.json.get('action')
    print('action from form is: {}'.format(action))
    state, reward, done, info = env.step(action)
    imutil.show(state, filename='static/screenshot.jpg')
    frame_num = env.env.unwrapped.ale.getFrameNumber()
    print('took action, now at frame num {}'.format(frame_num))
    if done:
        env.reset()
    #return flask.Response('OK', mimetype='application/json')
    #return flask.jsonify({'frame_num': frame_num})
    return 'OK'

@app.route('/static/<path:path>')
def route_static(path):
    return flask.send_from_directory('static', path)

if __name__ == '__main__':
    reset_game()
    app.run('0.0.0.0', debug=True, threaded=False)
