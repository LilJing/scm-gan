import flask
import imutil
import gym

app = flask.Flask(__name__)

def reset_game(name='Breakout-v0'):
    global env
    env = gym.make(name)
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
    frame_num = env.unwrapped.ale.getFrameNumber()
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
    app.run('0.0.0.0', debug=True)
