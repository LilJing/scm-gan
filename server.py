import flask
import imutil
import numpy as np
import atari_py

app = flask.Flask(__name__)

class GameEnv():
    def __init__(self, name):
        path = atari_py.get_game_path(name)
        self.ale = atari_py.ALEInterface()
        self.ale.loadROM(path)

    def reset(self):
        self.ale.reset_game()
        return self.ale.getScreenRGB2()

    def step(self, action):
        reward = self.ale.act(action)
        state = self.ale.getScreenRGB2()
        done = self.ale.game_over()
        info = {'ale.lives': self.ale.lives()}
        return state, reward, done, info


def reset_game():
    global env
    env = GameEnv('centipede')
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
    frame_num = env.ale.getFrameNumber()
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
