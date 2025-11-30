import os
import random
import time
import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from env_utils import create_environment, stack_frames, init_stacked_frames
from model import DDDQNNet
from memory import Memory
from config import (
    STATE_SIZE,
    TOTAL_EPISODES,
    MAX_STEPS,
    BATCH_SIZE,
    LEARNING_RATE,
    GAMMA,
    EXPLORE_START,
    EXPLORE_STOP,
    DECAY_RATE,
    MAX_TAU,
    PRETRAIN_LENGTH,
    MEMORY_SIZE,
    TRAINING,
    EPISODE_RENDER,
    MODEL_PATH,
    LOG_DIR,
)


def update_target_graph():
    """
    DQNetwork → TargetNetwork로 파라미터 복사하는 op 리스트 반환.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess, DQNetwork):
    """
    epsilon-greedy로 action을 결정.
    """
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if explore_probability > exp_exp_tradeoff:
        # explore
        choice = random.randrange(len(possible_actions))
        action = possible_actions[choice]
    else:
        # exploit
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


def main():
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

    game, possible_actions = create_environment()
    action_size = len(possible_actions)

    # 네트워크 생성
    tf.reset_default_graph()
    DQNetwork = DDDQNNet(STATE_SIZE, action_size, LEARNING_RATE, name="DQNetwork")
    TargetNetwork = DDDQNNet(STATE_SIZE, action_size, LEARNING_RATE, name="TargetNetwork")

    # 메모리
    memory = Memory(MEMORY_SIZE)

    # 초기 state/stack 설정
    stacked_frames = init_stacked_frames(STATE_SIZE[2])

    game.new_episode()
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size=STATE_SIZE[2])

    # pretrain: 무작위 policy로 경험 쌓기
    print("Filling replay memory with random gameplay...")
    for i in tqdm(range(PRETRAIN_LENGTH), desc="Pre-filling replay memory"):
        action = random.choice(possible_actions)
        reward = game.make_action(action)
        done = game.is_episode_finished()

        if done:
            next_state = np.zeros(state.shape)
            next_state_stacked = np.zeros(STATE_SIZE)
            memory.store(error=1.0, sample=(state, action, reward, next_state_stacked, done))

            game.new_episode()
            state = game.get_state().screen_buffer
            stacked_frames = init_stacked_frames(STATE_SIZE[2])
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size=STATE_SIZE[2])
        else:
            next_state = game.get_state().screen_buffer
            next_state_stacked, stacked_frames = stack_frames(
                stacked_frames, next_state, False, stack_size=STATE_SIZE[2]
            )
            memory.store(error=1.0, sample=(state, action, reward, next_state_stacked, done))
            state = next_state_stacked

        if (i + 1) % 10000 == 0:
            print(f"Filled {i+1}/{PRETRAIN_LENGTH} transitions")

    print("Replay memory filled. Start training...")

    # 타깃 네트워크 업데이트 op
    update_ops = update_target_graph()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(LOG_DIR)

    tf.summary.scalar("Loss", DQNetwork.loss)
    write_op = tf.summary.merge_all()

    global_step = 0
    decay_step = 0
    tau = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 처음 한 번 타깃 네트워크 동기화
        sess.run(update_ops)

        if not TRAINING and os.path.exists(MODEL_PATH + ".index"):
            print("Loading existing model...")
            saver.restore(sess, MODEL_PATH)

        for episode in range(TOTAL_EPISODES):
            game.new_episode()
            step = 0
            episode_reward = 0

            state = game.get_state().screen_buffer
            stacked_frames = init_stacked_frames(STATE_SIZE[2])
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size=STATE_SIZE[2])

            while step < MAX_STEPS:
                step += 1
                tau += 1
                global_step += 1

                if EPISODE_RENDER:
                    game.render()

                decay_step += 1
                action, explore_prob = predict_action(
                    EXPLORE_START, EXPLORE_STOP, DECAY_RATE,
                    decay_step, state, possible_actions, sess, DQNetwork
                )

                reward = game.make_action(action)
                done = game.is_episode_finished()
                episode_reward += reward

                if done:
                    next_state = np.zeros(state.shape)
                    next_state_stacked = np.zeros(STATE_SIZE)
                    memory.store(error=1.0, sample=(state, action, reward, next_state_stacked, done))
                    break
                else:
                    next_state = game.get_state().screen_buffer
                    next_state_stacked, stacked_frames = stack_frames(
                        stacked_frames, next_state, False, stack_size=STATE_SIZE[2]
                    )
                    memory.store(error=1.0, sample=(state, action, reward, next_state_stacked, done))
                    state = next_state_stacked

                t0 = time.time()
                # 학습
                tree_idx, batch, ISWeights = memory.sample(BATCH_SIZE)

                states_mb = np.array([each[0] for each in batch])
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch])
                dones_mb = np.array([each[4] for each in batch])

                # Double DQN target 계산
                Q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})
                best_actions = np.argmax(Q_next_state, axis=1)

                Q_target_next = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})
                target_Qs = rewards_mb + (1 - dones_mb) * GAMMA * Q_target_next[np.arange(BATCH_SIZE), best_actions]

                # one-hot action
                actions_one_hot = np.zeros((BATCH_SIZE, len(possible_actions)))
                for i, act in enumerate(actions_mb):
                    idx = possible_actions.index(list(act))
                    actions_one_hot[i, idx] = 1.0

                t1 = time.time()
                _, abs_errors, loss, summary = sess.run(
                    [DQNetwork.optimizer, DQNetwork.absolute_errors, DQNetwork.loss, write_op],
                    feed_dict={
                        DQNetwork.inputs_: states_mb,
                        DQNetwork.target_Q: target_Qs,
                        DQNetwork.actions_: actions_one_hot,
                        DQNetwork.ISWeights_: ISWeights,
                    },
                )
                t2 = time.time()

                # PER priority 업데이트
                memory.batch_update(tree_idx, abs_errors)

                if global_step % 100 == 0:
                    writer.add_summary(summary, global_step)
                    
                # 타깃 네트워크 일정 주기마다 동기화
                if tau > MAX_TAU:
                    sess.run(update_ops)
                    tau = 0

            writer.flush()
            print(f"Episode: {episode} Total reward: {episode_reward:.2f} Explore P: {explore_prob:.4f}")

            if TRAINING and (episode + 1) % 20 == 0:
                saver.save(sess, MODEL_PATH)
                print("Model saved at episode", episode + 1)

        if TRAINING:
            saver.save(sess, MODEL_PATH)
            print("Final model saved.")


if __name__ == "__main__":
    main()
