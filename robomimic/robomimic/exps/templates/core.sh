#!/bin/bash

# ==========core==========

#  task: lift
#    dataset type: ph
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/low_dim/iris.json

#  task: lift
#    dataset type: ph
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/ph/image/cql.json

#  task: lift
#    dataset type: mh
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/low_dim/iris.json

#  task: lift
#    dataset type: mh
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mh/image/cql.json

#  task: lift
#    dataset type: mg
#      hdf5 type: low_dim_sparse
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_sparse/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_sparse/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_sparse/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_sparse/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_sparse/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_sparse/iris.json

#  task: lift
#    dataset type: mg
#      hdf5 type: image_sparse
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/image_sparse/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/image_sparse/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/image_sparse/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/image_sparse/cql.json

#  task: lift
#    dataset type: mg
#      hdf5 type: low_dim_dense
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_dense/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_dense/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_dense/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_dense/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_dense/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/low_dim_dense/iris.json

#  task: lift
#    dataset type: mg
#      hdf5 type: image_dense
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/image_dense/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/image_dense/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/image_dense/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift/mg/image_dense/cql.json

#  task: can
#    dataset type: ph
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/low_dim/iris.json

#  task: can
#    dataset type: ph
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/ph/image/cql.json

#  task: can
#    dataset type: mh
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/low_dim/iris.json

#  task: can
#    dataset type: mh
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mh/image/cql.json

#  task: can
#    dataset type: mg
#      hdf5 type: low_dim_sparse
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_sparse/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_sparse/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_sparse/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_sparse/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_sparse/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_sparse/iris.json

#  task: can
#    dataset type: mg
#      hdf5 type: image_sparse
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/image_sparse/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/image_sparse/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/image_sparse/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/image_sparse/cql.json

#  task: can
#    dataset type: mg
#      hdf5 type: low_dim_dense
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_dense/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_dense/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_dense/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_dense/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_dense/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/low_dim_dense/iris.json

#  task: can
#    dataset type: mg
#      hdf5 type: image_dense
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/image_dense/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/image_dense/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/image_dense/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/mg/image_dense/cql.json

#  task: can
#    dataset type: paired
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/low_dim/iris.json

#  task: can
#    dataset type: paired
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can/paired/image/cql.json

#  task: square
#    dataset type: ph
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/low_dim/iris.json

#  task: square
#    dataset type: ph
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/ph/image/cql.json

#  task: square
#    dataset type: mh
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/low_dim/iris.json

#  task: square
#    dataset type: mh
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/square/mh/image/cql.json

#  task: transport
#    dataset type: ph
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/low_dim/iris.json

#  task: transport
#    dataset type: ph
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/ph/image/cql.json

#  task: transport
#    dataset type: mh
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/low_dim/iris.json

#  task: transport
#    dataset type: mh
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/transport/mh/image/cql.json

#  task: tool_hang
#    dataset type: ph
#      hdf5 type: low_dim
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/low_dim/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/low_dim/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/low_dim/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/low_dim/cql.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/low_dim/hbc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/low_dim/iris.json

#  task: tool_hang
#    dataset type: ph
#      hdf5 type: image
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/image/bc.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/image/bc_rnn.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/image/bcq.json
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang/ph/image/cql.json

#  task: lift_real
#    dataset type: ph
#      hdf5 type: raw
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/lift_real/ph/raw/bc_rnn.json

#  task: can_real
#    dataset type: ph
#      hdf5 type: raw
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/can_real/ph/raw/bc_rnn.json

#  task: tool_hang_real
#    dataset type: ph
#      hdf5 type: raw
python /data/home/prajj/code/cai_research/robomimic/robomimic/scripts/train.py --config robomimic/exps/templates/core/tool_hang_real/ph/raw/bc_rnn.json

