import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tqdm import tqdm
from env_utils import create_environment, stack_frames, init_stacked_frames
from model import DDDQNNet
from config import STATE_SIZE, MODEL_PATH


def main():
    game, possible_actions = create_environment()
    action_size = len(possible_actions)

    tf.reset_default_graph()
    DQNetwork = DDDQNNet(STATE_SIZE, action_size, learning_rate=0.00025, name="DQNetwork")

    saver = tf.train.Saver()

    stacked_frames = init_stacked_frames(STATE_SIZE[2])

    with tf.Session() as sess:
        game = create_environment()[0]

        # 테스트용 cfg / wad 사용하고 싶으면 여기를 수정
        game.load_config("deadly_corridor.cfg")
        game.set_doom_scenario_path("deadly_corridor.wad")
        game.init()

        saver.restore(sess, MODEL_PATH)
        print("Model restored. Watching agent play...")

        for i in range(10):
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size=STATE_SIZE[2])

            while not game.is_episode_finished():
                Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]

                game.make_action(action)

                if game.is_episode_finished():
                    break

                next_state = game.get_state().screen_buffer
                state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size=STATE_SIZE[2])

            score = game.get_total_reward()
            print(f"Episode {i} score: {score}")

        game.close()


if __name__ == "__main__":
    main()
