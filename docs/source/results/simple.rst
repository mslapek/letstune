Simple results
==============

Summary
<<<<<<<

.. automodule:: letstune.results.simple

Tuning results class
<<<<<<<<<<<<<<<<<<<<

.. autoclass:: letstune.results.simple.TuningResults
   :members: to_df, metric, metric_value

   Bases: :class:`typing.Generic` [``P``], :class:`collections.abc.Sequence` [:class:`letstune.results.simple.Training` [``P``]]

   .. method:: __getitem__(i: int) -> letstune.results.simple.Training[P]

      Get *i*-th best training. ``tuning[0]`` gives the best training.

   .. method:: __getitem__(slice: slice) -> list[letstune.results.simple.Training[P]]
      :noindex:

      Get slice of top trainings. ``tuning[:5]`` gives 5 best trainings.

   .. method:: __len__() -> int

      Get number of trainings.

   .. property:: errors
      :type: collections.abc.Sequence[letstune.results.simple.Error]

      Sequence of failed trainings.


Other classes
<<<<<<<<<<<<<

.. autoclass:: letstune.results.simple.Training

   Bases: :class:`typing.Generic` [ ``P`` ]

   **Basic data**

   .. autoattribute:: training_id
   .. attribute:: params
      :type: P

   **Metric values**

   .. autoattribute:: metric_values
   .. autoattribute:: metric_value

   **Time**

   .. autoattribute:: start_time
   .. autoattribute:: end_time
   .. autoproperty:: duration

   **Checkpoint**

   .. autoproperty:: checkpoint

.. autoclass:: letstune.results.simple.Error

   .. autoattribute:: training_id
   .. autoattribute:: params
   .. autoattribute:: msg
