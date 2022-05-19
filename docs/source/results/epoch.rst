Epoch results
=============

Summary
<<<<<<<

.. automodule:: letstune.results.epoch

Tuning results class
<<<<<<<<<<<<<<<<<<<<

.. autoclass:: letstune.results.epoch.TuningResults
   :members: to_df, metric, metric_value

   Bases: :class:`typing.Generic` [``P``], :class:`collections.abc.Sequence` [:class:`letstune.results.epoch.Training` [``P``]]

   .. method:: __getitem__(i: int) -> letstune.results.epoch.Training[P]

      Get *i*-th best training. ``tuning[0]`` gives the best training.

   .. method:: __getitem__(slice: slice) -> list[letstune.results.epoch.Training[P]]
      :noindex:

      Get slice of top trainings. ``tuning[:5]`` gives 5 best trainings.

   .. method:: __len__() -> int

      Get number of trainings.

   .. property:: errors
      :type: collections.abc.Sequence[letstune.results.epoch.Error]

      Sequence of failed trainings.


Other classes
<<<<<<<<<<<<<


.. autoclass:: letstune.results.epoch.Training

   Bases: :class:`typing.Generic` [ ``P`` ], :class:`collections.abc.Sequence` [ :class:`letstune.results.epoch.Epoch` ]

   **Basic data**

   .. autoattribute:: training_id
   .. attribute:: params
      :type: P

   .. autoattribute:: round

      Round survived by the training. Calculated using :attr:`metric_value`.

   **Epochs**

   .. attribute:: best_epoch
      :type: Epoch

      The best epoch in the training.

      Can be different than :attr:`last_epoch` due to
      overfit.

   .. attribute:: last_epoch
      :type: Epoch

      Last epoch in the training.

   .. method:: __getitem__(i: int) -> letstune.results.epoch.Epoch

      Get *i*-th epoch.

   .. method:: __len__() -> int

      Get number of epochs.

   **Summarized metric values**

   .. autoproperty:: metric_value

   **Summarized time properties**

   .. autoproperty:: start_time
   .. autoproperty:: end_time
   .. autoproperty:: duration

.. autoclass:: letstune.results.epoch.Epoch

   **Basic data**

   .. autoattribute:: training_id
   .. autoattribute:: epoch_id

   **Metric values**

   .. autoattribute:: metric_values
   .. autoattribute:: metric_value
   .. autoattribute:: total_metric_value

      The best metric value up to this epoch in this training.

   **Time**

   .. autoattribute:: start_time
   .. autoattribute:: end_time
   .. autoproperty:: duration
   .. autoattribute:: total_duration

      Total training duration up to this epoch.

   **Checkpoint**

   .. autoproperty:: checkpoint

.. autoclass:: letstune.results.epoch.Error

   .. autoattribute:: training_id
   .. autoattribute:: params
   .. autoattribute:: msg
