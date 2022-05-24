.. _trainers:

Trainers
========

Summary
<<<<<<<

.. automodule:: letstune.trainer

Metric class
<<<<<<<<<<<<

.. autoclass:: letstune.Metric
   :members:

Simple trainer class
<<<<<<<<<<<<<<<<<<<<

.. autoclass:: letstune.SimpleTrainer

   Bases: :class:`typing.Generic` [``P``], :class:`abc.ABC`

   **Obligatory methods**

   Methods, which must be implemented in a trainer:

   .. autoproperty:: metric
   .. automethod:: load_dataset
   .. automethod:: train

   **Optional methods**

   Methods, which can me overridden for further customization:

   .. automethod:: save
   .. automethod:: get_random_params

Epoch trainer class
<<<<<<<<<<<<<<<<<<<<

.. autoclass:: letstune.EpochTrainer

   Bases: :class:`typing.Generic` [``P``], :class:`abc.ABC`

   **Obligatory methods**

   Methods, which must be implemented in a trainer.

   .. autoproperty:: metric
   .. automethod:: load_dataset

   **Lifecycle methods**

   .. automethod:: create_model
   .. automethod:: train_epoch
   .. automethod:: save
   .. automethod:: load

   **Miscellaneous**

   Methods, which can me overridden for further customization:

   .. automethod:: get_random_params


.. _rounds:

Investment rounds
<<<<<<<<<<<<<<<<<

When training with :class:`letstune.EpochTrainer`,
*letstune* spends most of the time on the most promising
parameters.

*letstune* makes a kind of investment rounds.

At the first round, it evaluates all parameters for a few epochs.

Only 25% of trainings will advance to the next round.
Trainings with the lowest metric value are automatically dropped.
