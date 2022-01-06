#!/bin/bash

source ${SOURCE_DIR}/run_scripts/run.sh

if [[ "$1" = "test" ]]; then
	WEIGHTS=${WEIGHTS:-checkpoint/model_best_eval.pth}
	TEST_PARAMS="mode=test model.weights=$WEIGHTS $EXTRA_TEST_PARAMS"
fi

$PYTHON_CMD ${SOURCE_DIR}/main.py \
	name=dconv \
	experiment=saliency \
	train.precision=16 \
	data.batch_size=2 \
	model.inter_block=GC3d \
	model.refine_block=Refine3d \
	model.decoder.norm_type=GroupNorm \
	model.decoder.norm_groups=1 \
	model.decoder.interpolation_mode=trilinear \
	model.decoder.align_corners=True \
	train.num_epochs=20 \
	model/decoder/deformable=DCNv1 \
	model.decoder.deformable.layers=[rf2,rf3] \
	model/backbone=irCSN152 \
	model.backbone.replace_norm_layer=True \
	model.backbone.norm_layer=FrozenBatchNorm \
	model.backbone.norm_layer_eps=1e-3 \
	loss.name=lovasz_ce \
	train.grad_clip_type=norm \
	train.grad_clip_max=10 \
	data.eval.imset=2016 \
	${TEST_PARAMS}
