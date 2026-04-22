# Game Design

## Core Fantasy

The player uses hand gestures to slice or hit flying food items, similar to Fruit Ninja, while learning the three Palau food groups.

## Educational Goal

Teach players to recognize:

1. Protective Food Group
   Fruits and vegetables that help protect the body from illness and disease.
2. Energy Food Group
   Foods rich in carbohydrates and fiber that provide energy.
3. Body Building Foods Group
   Foods rich in protein that help repair and build body tissues.

## Throw and Spawn Design

Food should be thrown in arcs from the lower half of the screen, with timing inspired by Fruit Ninja.

### Spawn Principles

- Spawn in short waves rather than at a constant interval.
- Mix easy and difficult trajectories.
- Keep the center of the screen active so swipes feel satisfying.
- Avoid overlapping too many items at once on low-power devices.

### Recommended Spawn Behaviors

- Single throw:
  one item with a clean arc for beginner rounds
- Pair throw:
  two items with mirrored or slightly offset arcs
- Group burst:
  three to five items with varied horizontal velocity
- Quiz throw:
  one target food group item plus distractors from other groups

### Difficulty Variables

- vertical launch speed
- horizontal launch speed
- burst size
- delay between waves
- target group emphasis

## Example Round Loop

1. Show the current learning prompt.
2. Spawn a burst of foods.
3. Track hand motion and interpret swipes.
4. Evaluate hit accuracy and food group correctness.
5. Award score and brief feedback.
6. Move to the next wave.

## Future Expansion

- voice prompts
- group-specific mini-challenges
- timed survival mode
- class mode with level progression
