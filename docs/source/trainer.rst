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

