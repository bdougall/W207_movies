??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
embedding_8/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*'
shared_nameembedding_8/embeddings
?
*embedding_8/embeddings/Read/ReadVariableOpReadVariableOpembedding_8/embeddings* 
_output_shapes
:
??
*
dtype0
?
embedding_9/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*'
shared_nameembedding_9/embeddings
?
*embedding_9/embeddings/Read/ReadVariableOpReadVariableOpembedding_9/embeddings* 
_output_shapes
:
??
*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:?*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	?*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/embedding_8/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*.
shared_nameAdam/embedding_8/embeddings/m
?
1Adam/embedding_8/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_8/embeddings/m* 
_output_shapes
:
??
*
dtype0
?
Adam/embedding_9/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*.
shared_nameAdam/embedding_9/embeddings/m
?
1Adam/embedding_9/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_9/embeddings/m* 
_output_shapes
:
??
*
dtype0
?
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/m
?
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	?*
dtype0

Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_8/bias/m
x
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_9/kernel/m
?
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_8/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*.
shared_nameAdam/embedding_8/embeddings/v
?
1Adam/embedding_8/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_8/embeddings/v* 
_output_shapes
:
??
*
dtype0
?
Adam/embedding_9/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*.
shared_nameAdam/embedding_9/embeddings/v
?
1Adam/embedding_9/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_9/embeddings/v* 
_output_shapes
:
??
*
dtype0
?
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/v
?
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	?*
dtype0

Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_8/bias/v
x
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_9/kernel/v
?
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?-
value?-B?- B?-
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures

_init_input_shape

_init_input_shape
b

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
b

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
 trainable_variables
!regularization_losses
"	variables
#	keras_api
R
$trainable_variables
%regularization_losses
&	variables
'	keras_api
h

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
h

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
?
4iter

5beta_1

6beta_2
	7decay
8learning_ratemfmg(mh)mi.mj/mkvlvm(vn)vo.vp/vq
 
*
0
1
(2
)3
.4
/5
*
0
1
(2
)3
.4
/5
?

9layers
regularization_losses
trainable_variables
:non_trainable_variables
;layer_metrics
	variables
<metrics
=layer_regularization_losses
 
 
 
fd
VARIABLE_VALUEembedding_8/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?

>layers
trainable_variables
regularization_losses
?non_trainable_variables
@layer_metrics
	variables
Ametrics
Blayer_regularization_losses
fd
VARIABLE_VALUEembedding_9/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?

Clayers
trainable_variables
regularization_losses
Dnon_trainable_variables
Elayer_metrics
	variables
Fmetrics
Glayer_regularization_losses
 
 
 
?

Hlayers
trainable_variables
regularization_losses
Inon_trainable_variables
Jlayer_metrics
	variables
Kmetrics
Llayer_regularization_losses
 
 
 
?

Mlayers
 trainable_variables
!regularization_losses
Nnon_trainable_variables
Olayer_metrics
"	variables
Pmetrics
Qlayer_regularization_losses
 
 
 
?

Rlayers
$trainable_variables
%regularization_losses
Snon_trainable_variables
Tlayer_metrics
&	variables
Umetrics
Vlayer_regularization_losses
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?

Wlayers
*trainable_variables
+regularization_losses
Xnon_trainable_variables
Ylayer_metrics
,	variables
Zmetrics
[layer_regularization_losses
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?

\layers
0trainable_variables
1regularization_losses
]non_trainable_variables
^layer_metrics
2	variables
_metrics
`layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8
 
 

a0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	btotal
	ccount
d	variables
e	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

b0
c1

d	variables
??
VARIABLE_VALUEAdam/embedding_8/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_9/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_8/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_9/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_10Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_9Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10serving_default_input_9embedding_9/embeddingsembedding_8/embeddingsdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8? */
f*R(
&__inference_signature_wrapper_10981972
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_8/embeddings/Read/ReadVariableOp*embedding_9/embeddings/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/embedding_8/embeddings/m/Read/ReadVariableOp1Adam/embedding_9/embeddings/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp1Adam/embedding_8/embeddings/v/Read/ReadVariableOp1Adam/embedding_9/embeddings/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? **
f%R#
!__inference__traced_save_10982290
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_8/embeddingsembedding_9/embeddingsdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/embedding_8/embeddings/mAdam/embedding_9/embeddings/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/embedding_8/embeddings/vAdam/embedding_9/embeddings/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *-
f(R&
$__inference__traced_restore_10982375ݓ
?
?
*__inference_dense_8_layer_call_fn_10982171

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_109817262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_9_layer_call_and_return_conditional_losses_10982133

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????
2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
*__inference_dense_9_layer_call_fn_10982191

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_109817432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
E__inference_model_4_layer_call_and_return_conditional_losses_10982009
inputs_0
inputs_19
%embedding_9_embedding_lookup_10981977:
??
9
%embedding_8_embedding_lookup_10981983:
??
9
&dense_8_matmul_readvariableop_resource:	?6
'dense_8_biasadd_readvariableop_resource:	?9
&dense_9_matmul_readvariableop_resource:	?5
'dense_9_biasadd_readvariableop_resource:
identity??dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?embedding_8/embedding_lookup?embedding_9/embedding_lookupw
embedding_9/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_9/Cast?
embedding_9/embedding_lookupResourceGather%embedding_9_embedding_lookup_10981977embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_9/embedding_lookup/10981977*+
_output_shapes
:?????????
*
dtype02
embedding_9/embedding_lookup?
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_9/embedding_lookup/10981977*+
_output_shapes
:?????????
2'
%embedding_9/embedding_lookup/Identity?
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2)
'embedding_9/embedding_lookup/Identity_1w
embedding_8/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_8/Cast?
embedding_8/embedding_lookupResourceGather%embedding_8_embedding_lookup_10981983embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_8/embedding_lookup/10981983*+
_output_shapes
:?????????
*
dtype02
embedding_8/embedding_lookup?
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_8/embedding_lookup/10981983*+
_output_shapes
:?????????
2'
%embedding_8/embedding_lookup/Identity?
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2)
'embedding_8/embedding_lookup/Identity_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
flatten_8/Const?
flatten_8/ReshapeReshape0embedding_8/embedding_lookup/Identity_1:output:0flatten_8/Const:output:0*
T0*'
_output_shapes
:?????????
2
flatten_8/Reshapes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
flatten_9/Const?
flatten_9/ReshapeReshape0embedding_9/embedding_lookup/Identity_1:output:0flatten_9/Const:output:0*
T0*'
_output_shapes
:?????????
2
flatten_9/Reshapex
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2flatten_8/Reshape:output:0flatten_9/Reshape:output:0"concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_4/concat?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulconcatenate_4/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_8/Relu?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Relu?
IdentityIdentitydense_9/Relu:activations:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp^embedding_8/embedding_lookup^embedding_9/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2<
embedding_8/embedding_lookupembedding_8/embedding_lookup2<
embedding_9/embedding_lookupembedding_9/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
I__inference_embedding_8_layer_call_and_return_conditional_losses_10981686

inputs-
embedding_lookup_10981680:
??

identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10981680Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10981680*+
_output_shapes
:?????????
*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10981680*+
_output_shapes
:?????????
2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
0__inference_concatenate_4_layer_call_fn_10982151
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_109817132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????
:?????????
:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1
?	
?
*__inference_model_4_layer_call_fn_10982064
inputs_0
inputs_1
unknown:
??

	unknown_0:
??

	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_109817502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
w
K__inference_concatenate_4_layer_call_and_return_conditional_losses_10982145
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????
:?????????
:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1
?
?
.__inference_embedding_8_layer_call_fn_10982099

inputs
unknown:
??

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_8_layer_call_and_return_conditional_losses_109816862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_model_4_layer_call_fn_10982082
inputs_0
inputs_1
unknown:
??

	unknown_0:
??

	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_109818652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
c
G__inference_flatten_8_layer_call_and_return_conditional_losses_10982122

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????
2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?!
?
E__inference_model_4_layer_call_and_return_conditional_losses_10981922
input_9
input_10(
embedding_9_10981902:
??
(
embedding_8_10981905:
??
#
dense_8_10981911:	?
dense_8_10981913:	?#
dense_9_10981916:	?
dense_9_10981918:
identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?#embedding_8/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallinput_10embedding_9_10981902*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_9_layer_call_and_return_conditional_losses_109816722%
#embedding_9/StatefulPartitionedCall?
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_9embedding_8_10981905*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_8_layer_call_and_return_conditional_losses_109816862%
#embedding_8/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall,embedding_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_109816962
flatten_8/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall,embedding_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_109817042
flatten_9/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_109817132
concatenate_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_8_10981911dense_8_10981913*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_109817262!
dense_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_10981916dense_9_10981918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_109817432!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_9:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?m
?
$__inference__traced_restore_10982375
file_prefix;
'assignvariableop_embedding_8_embeddings:
??
=
)assignvariableop_1_embedding_9_embeddings:
??
4
!assignvariableop_2_dense_8_kernel:	?.
assignvariableop_3_dense_8_bias:	?4
!assignvariableop_4_dense_9_kernel:	?-
assignvariableop_5_dense_9_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: E
1assignvariableop_13_adam_embedding_8_embeddings_m:
??
E
1assignvariableop_14_adam_embedding_9_embeddings_m:
??
<
)assignvariableop_15_adam_dense_8_kernel_m:	?6
'assignvariableop_16_adam_dense_8_bias_m:	?<
)assignvariableop_17_adam_dense_9_kernel_m:	?5
'assignvariableop_18_adam_dense_9_bias_m:E
1assignvariableop_19_adam_embedding_8_embeddings_v:
??
E
1assignvariableop_20_adam_embedding_9_embeddings_v:
??
<
)assignvariableop_21_adam_dense_8_kernel_v:	?6
'assignvariableop_22_adam_dense_8_bias_v:	?<
)assignvariableop_23_adam_dense_9_kernel_v:	?5
'assignvariableop_24_adam_dense_9_bias_v:
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_8_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_9_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_8_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_8_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_9_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_9_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp1assignvariableop_13_adam_embedding_8_embeddings_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp1assignvariableop_14_adam_embedding_9_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_8_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_8_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_9_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_9_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_embedding_8_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_embedding_9_embeddings_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_8_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_8_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_9_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_9_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25?
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
*__inference_model_4_layer_call_fn_10981765
input_9
input_10
unknown:
??

	unknown_0:
??

	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_109817502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_9:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?	
?
&__inference_signature_wrapper_10981972
input_10
input_9
unknown:
??

	unknown_0:
??

	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8? *,
f'R%
#__inference__wrapped_model_109816532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_9
?!
?
E__inference_model_4_layer_call_and_return_conditional_losses_10981946
input_9
input_10(
embedding_9_10981926:
??
(
embedding_8_10981929:
??
#
dense_8_10981935:	?
dense_8_10981937:	?#
dense_9_10981940:	?
dense_9_10981942:
identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?#embedding_8/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallinput_10embedding_9_10981926*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_9_layer_call_and_return_conditional_losses_109816722%
#embedding_9/StatefulPartitionedCall?
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_9embedding_8_10981929*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_8_layer_call_and_return_conditional_losses_109816862%
#embedding_8/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall,embedding_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_109816962
flatten_8/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall,embedding_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_109817042
flatten_9/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_109817132
concatenate_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_8_10981935dense_8_10981937*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_109817262!
dense_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_10981940dense_9_10981942*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_109817432!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_9:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?
H
,__inference_flatten_8_layer_call_fn_10982127

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_109816962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
c
G__inference_flatten_8_layer_call_and_return_conditional_losses_10981696

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????
2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
I__inference_embedding_9_layer_call_and_return_conditional_losses_10981672

inputs-
embedding_lookup_10981666:
??

identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10981666Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10981666*+
_output_shapes
:?????????
*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10981666*+
_output_shapes
:?????????
2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_model_4_layer_call_fn_10981898
input_9
input_10
unknown:
??

	unknown_0:
??

	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_109818652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_9:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?

?
I__inference_embedding_8_layer_call_and_return_conditional_losses_10982092

inputs-
embedding_lookup_10982086:
??

identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10982086Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10982086*+
_output_shapes
:?????????
*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10982086*+
_output_shapes
:?????????
2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_9_layer_call_and_return_conditional_losses_10981704

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????
2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?!
?
E__inference_model_4_layer_call_and_return_conditional_losses_10981750

inputs
inputs_1(
embedding_9_10981673:
??
(
embedding_8_10981687:
??
#
dense_8_10981727:	?
dense_8_10981729:	?#
dense_9_10981744:	?
dense_9_10981746:
identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?#embedding_8/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_9_10981673*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_9_layer_call_and_return_conditional_losses_109816722%
#embedding_9/StatefulPartitionedCall?
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_8_10981687*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_8_layer_call_and_return_conditional_losses_109816862%
#embedding_8/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall,embedding_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_109816962
flatten_8/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall,embedding_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_109817042
flatten_9/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_109817132
concatenate_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_8_10981727dense_8_10981729*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_109817262!
dense_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_10981744dense_9_10981746*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_109817432!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_9_layer_call_and_return_conditional_losses_10981743

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_8_layer_call_and_return_conditional_losses_10982162

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?

!__inference__traced_save_10982290
file_prefix5
1savev2_embedding_8_embeddings_read_readvariableop5
1savev2_embedding_9_embeddings_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_embedding_8_embeddings_m_read_readvariableop<
8savev2_adam_embedding_9_embeddings_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop<
8savev2_adam_embedding_8_embeddings_v_read_readvariableop<
8savev2_adam_embedding_9_embeddings_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_8_embeddings_read_readvariableop1savev2_embedding_9_embeddings_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_embedding_8_embeddings_m_read_readvariableop8savev2_adam_embedding_9_embeddings_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop8savev2_adam_embedding_8_embeddings_v_read_readvariableop8savev2_adam_embedding_9_embeddings_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??
:
??
:	?:?:	?:: : : : : : : :
??
:
??
:	?:?:	?::
??
:
??
:	?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??
:&"
 
_output_shapes
:
??
:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??
:&"
 
_output_shapes
:
??
:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
??
:&"
 
_output_shapes
:
??
:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?5
?
#__inference__wrapped_model_10981653
input_9
input_10A
-model_4_embedding_9_embedding_lookup_10981621:
??
A
-model_4_embedding_8_embedding_lookup_10981627:
??
A
.model_4_dense_8_matmul_readvariableop_resource:	?>
/model_4_dense_8_biasadd_readvariableop_resource:	?A
.model_4_dense_9_matmul_readvariableop_resource:	?=
/model_4_dense_9_biasadd_readvariableop_resource:
identity??&model_4/dense_8/BiasAdd/ReadVariableOp?%model_4/dense_8/MatMul/ReadVariableOp?&model_4/dense_9/BiasAdd/ReadVariableOp?%model_4/dense_9/MatMul/ReadVariableOp?$model_4/embedding_8/embedding_lookup?$model_4/embedding_9/embedding_lookup?
model_4/embedding_9/CastCastinput_10*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_4/embedding_9/Cast?
$model_4/embedding_9/embedding_lookupResourceGather-model_4_embedding_9_embedding_lookup_10981621model_4/embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*@
_class6
42loc:@model_4/embedding_9/embedding_lookup/10981621*+
_output_shapes
:?????????
*
dtype02&
$model_4/embedding_9/embedding_lookup?
-model_4/embedding_9/embedding_lookup/IdentityIdentity-model_4/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@model_4/embedding_9/embedding_lookup/10981621*+
_output_shapes
:?????????
2/
-model_4/embedding_9/embedding_lookup/Identity?
/model_4/embedding_9/embedding_lookup/Identity_1Identity6model_4/embedding_9/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
21
/model_4/embedding_9/embedding_lookup/Identity_1?
model_4/embedding_8/CastCastinput_9*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_4/embedding_8/Cast?
$model_4/embedding_8/embedding_lookupResourceGather-model_4_embedding_8_embedding_lookup_10981627model_4/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*@
_class6
42loc:@model_4/embedding_8/embedding_lookup/10981627*+
_output_shapes
:?????????
*
dtype02&
$model_4/embedding_8/embedding_lookup?
-model_4/embedding_8/embedding_lookup/IdentityIdentity-model_4/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@model_4/embedding_8/embedding_lookup/10981627*+
_output_shapes
:?????????
2/
-model_4/embedding_8/embedding_lookup/Identity?
/model_4/embedding_8/embedding_lookup/Identity_1Identity6model_4/embedding_8/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
21
/model_4/embedding_8/embedding_lookup/Identity_1?
model_4/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
model_4/flatten_8/Const?
model_4/flatten_8/ReshapeReshape8model_4/embedding_8/embedding_lookup/Identity_1:output:0 model_4/flatten_8/Const:output:0*
T0*'
_output_shapes
:?????????
2
model_4/flatten_8/Reshape?
model_4/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
model_4/flatten_9/Const?
model_4/flatten_9/ReshapeReshape8model_4/embedding_9/embedding_lookup/Identity_1:output:0 model_4/flatten_9/Const:output:0*
T0*'
_output_shapes
:?????????
2
model_4/flatten_9/Reshape?
!model_4/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_4/concatenate_4/concat/axis?
model_4/concatenate_4/concatConcatV2"model_4/flatten_8/Reshape:output:0"model_4/flatten_9/Reshape:output:0*model_4/concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model_4/concatenate_4/concat?
%model_4/dense_8/MatMul/ReadVariableOpReadVariableOp.model_4_dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%model_4/dense_8/MatMul/ReadVariableOp?
model_4/dense_8/MatMulMatMul%model_4/concatenate_4/concat:output:0-model_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_8/MatMul?
&model_4/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_4/dense_8/BiasAdd/ReadVariableOp?
model_4/dense_8/BiasAddBiasAdd model_4/dense_8/MatMul:product:0.model_4/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_8/BiasAdd?
model_4/dense_8/ReluRelu model_4/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_4/dense_8/Relu?
%model_4/dense_9/MatMul/ReadVariableOpReadVariableOp.model_4_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%model_4/dense_9/MatMul/ReadVariableOp?
model_4/dense_9/MatMulMatMul"model_4/dense_8/Relu:activations:0-model_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_9/MatMul?
&model_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_4/dense_9/BiasAdd/ReadVariableOp?
model_4/dense_9/BiasAddBiasAdd model_4/dense_9/MatMul:product:0.model_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_9/BiasAdd?
model_4/dense_9/ReluRelu model_4/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/dense_9/Relu?
IdentityIdentity"model_4/dense_9/Relu:activations:0'^model_4/dense_8/BiasAdd/ReadVariableOp&^model_4/dense_8/MatMul/ReadVariableOp'^model_4/dense_9/BiasAdd/ReadVariableOp&^model_4/dense_9/MatMul/ReadVariableOp%^model_4/embedding_8/embedding_lookup%^model_4/embedding_9/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2P
&model_4/dense_8/BiasAdd/ReadVariableOp&model_4/dense_8/BiasAdd/ReadVariableOp2N
%model_4/dense_8/MatMul/ReadVariableOp%model_4/dense_8/MatMul/ReadVariableOp2P
&model_4/dense_9/BiasAdd/ReadVariableOp&model_4/dense_9/BiasAdd/ReadVariableOp2N
%model_4/dense_9/MatMul/ReadVariableOp%model_4/dense_9/MatMul/ReadVariableOp2L
$model_4/embedding_8/embedding_lookup$model_4/embedding_8/embedding_lookup2L
$model_4/embedding_9/embedding_lookup$model_4/embedding_9/embedding_lookup:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_9:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?!
?
E__inference_model_4_layer_call_and_return_conditional_losses_10981865

inputs
inputs_1(
embedding_9_10981845:
??
(
embedding_8_10981848:
??
#
dense_8_10981854:	?
dense_8_10981856:	?#
dense_9_10981859:	?
dense_9_10981861:
identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?#embedding_8/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_9_10981845*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_9_layer_call_and_return_conditional_losses_109816722%
#embedding_9/StatefulPartitionedCall?
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_8_10981848*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_8_layer_call_and_return_conditional_losses_109816862%
#embedding_8/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall,embedding_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_109816962
flatten_8/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall,embedding_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_109817042
flatten_9/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_109817132
concatenate_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_8_10981854dense_8_10981856*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_109817262!
dense_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_10981859dense_9_10981861*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_109817432!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
u
K__inference_concatenate_4_layer_call_and_return_conditional_losses_10981713

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
I__inference_embedding_9_layer_call_and_return_conditional_losses_10982109

inputs-
embedding_lookup_10982103:
??

identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10982103Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10982103*+
_output_shapes
:?????????
*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10982103*+
_output_shapes
:?????????
2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_8_layer_call_and_return_conditional_losses_10981726

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_9_layer_call_fn_10982138

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_109817042
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
.__inference_embedding_9_layer_call_fn_10982116

inputs
unknown:
??

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *R
fMRK
I__inference_embedding_9_layer_call_and_return_conditional_losses_109816722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
E__inference_model_4_layer_call_and_return_conditional_losses_10982046
inputs_0
inputs_19
%embedding_9_embedding_lookup_10982014:
??
9
%embedding_8_embedding_lookup_10982020:
??
9
&dense_8_matmul_readvariableop_resource:	?6
'dense_8_biasadd_readvariableop_resource:	?9
&dense_9_matmul_readvariableop_resource:	?5
'dense_9_biasadd_readvariableop_resource:
identity??dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?embedding_8/embedding_lookup?embedding_9/embedding_lookupw
embedding_9/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_9/Cast?
embedding_9/embedding_lookupResourceGather%embedding_9_embedding_lookup_10982014embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_9/embedding_lookup/10982014*+
_output_shapes
:?????????
*
dtype02
embedding_9/embedding_lookup?
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_9/embedding_lookup/10982014*+
_output_shapes
:?????????
2'
%embedding_9/embedding_lookup/Identity?
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2)
'embedding_9/embedding_lookup/Identity_1w
embedding_8/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_8/Cast?
embedding_8/embedding_lookupResourceGather%embedding_8_embedding_lookup_10982020embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_8/embedding_lookup/10982020*+
_output_shapes
:?????????
*
dtype02
embedding_8/embedding_lookup?
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_8/embedding_lookup/10982020*+
_output_shapes
:?????????
2'
%embedding_8/embedding_lookup/Identity?
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2)
'embedding_8/embedding_lookup/Identity_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
flatten_8/Const?
flatten_8/ReshapeReshape0embedding_8/embedding_lookup/Identity_1:output:0flatten_8/Const:output:0*
T0*'
_output_shapes
:?????????
2
flatten_8/Reshapes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
flatten_9/Const?
flatten_9/ReshapeReshape0embedding_9/embedding_lookup/Identity_1:output:0flatten_9/Const:output:0*
T0*'
_output_shapes
:?????????
2
flatten_9/Reshapex
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2flatten_8/Reshape:output:0flatten_9/Reshape:output:0"concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_4/concat?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulconcatenate_4/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_8/Relu?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Relu?
IdentityIdentitydense_9/Relu:activations:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp^embedding_8/embedding_lookup^embedding_9/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2<
embedding_8/embedding_lookupembedding_8/embedding_lookup2<
embedding_9/embedding_lookupembedding_9/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
E__inference_dense_9_layer_call_and_return_conditional_losses_10982182

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_101
serving_default_input_10:0?????????
;
input_90
serving_default_input_9:0?????????;
dense_90
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?D
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*r&call_and_return_all_conditional_losses
s__call__
t_default_save_signature"?A
_tf_keras_network?A{"name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 44976, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_8", "inbound_nodes": [[["input_9", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 136363, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_9", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["embedding_8", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["embedding_9", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["flatten_8", 0, 0, {}], ["flatten_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}], "input_layers": [["input_9", 0, 0], ["input_10", 0, 0]], "output_layers": [["dense_9", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "input_9"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "input_10"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 44976, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_8", "inbound_nodes": [[["input_9", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Embedding", "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 136363, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_9", "inbound_nodes": [[["input_10", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["embedding_8", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["embedding_9", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["flatten_8", 0, 0, {}], ["flatten_9", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["input_9", 0, 0], ["input_10", 0, 0]], "output_layers": [["dense_9", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
_init_input_shape"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_9", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}
?
_init_input_shape"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_10", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}}
?

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
*u&call_and_return_all_conditional_losses
v__call__"?
_tf_keras_layer?{"name": "embedding_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 44976, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["input_9", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"?
_tf_keras_layer?{"name": "embedding_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 136363, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["input_10", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"?
_tf_keras_layer?{"name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["embedding_8", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
?
 trainable_variables
!regularization_losses
"	variables
#	keras_api
*{&call_and_return_all_conditional_losses
|__call__"?
_tf_keras_layer?{"name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["embedding_9", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 19}}
?
$trainable_variables
%regularization_losses
&	variables
'	keras_api
*}&call_and_return_all_conditional_losses
~__call__"?
_tf_keras_layer?{"name": "concatenate_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["flatten_8", 0, 0, {}], ["flatten_9", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10]}, {"class_name": "TensorShape", "items": [null, 10]}]}
?	

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
*&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_4", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_8", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
4iter

5beta_1

6beta_2
	7decay
8learning_ratemfmg(mh)mi.mj/mkvlvm(vn)vo.vp/vq"
	optimizer
 "
trackable_list_wrapper
J
0
1
(2
)3
.4
/5"
trackable_list_wrapper
J
0
1
(2
)3
.4
/5"
trackable_list_wrapper
?

9layers
regularization_losses
trainable_variables
:non_trainable_variables
;layer_metrics
	variables
<metrics
=layer_regularization_losses
s__call__
t_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
*:(
??
2embedding_8/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?

>layers
trainable_variables
regularization_losses
?non_trainable_variables
@layer_metrics
	variables
Ametrics
Blayer_regularization_losses
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
*:(
??
2embedding_9/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?

Clayers
trainable_variables
regularization_losses
Dnon_trainable_variables
Elayer_metrics
	variables
Fmetrics
Glayer_regularization_losses
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Hlayers
trainable_variables
regularization_losses
Inon_trainable_variables
Jlayer_metrics
	variables
Kmetrics
Llayer_regularization_losses
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Mlayers
 trainable_variables
!regularization_losses
Nnon_trainable_variables
Olayer_metrics
"	variables
Pmetrics
Qlayer_regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Rlayers
$trainable_variables
%regularization_losses
Snon_trainable_variables
Tlayer_metrics
&	variables
Umetrics
Vlayer_regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_8/kernel
:?2dense_8/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?

Wlayers
*trainable_variables
+regularization_losses
Xnon_trainable_variables
Ylayer_metrics
,	variables
Zmetrics
[layer_regularization_losses
?__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_9/kernel
:2dense_9/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?

\layers
0trainable_variables
1regularization_losses
]non_trainable_variables
^layer_metrics
2	variables
_metrics
`layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
a0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	btotal
	ccount
d	variables
e	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 22}
:  (2total
:  (2count
.
b0
c1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
/:-
??
2Adam/embedding_8/embeddings/m
/:-
??
2Adam/embedding_9/embeddings/m
&:$	?2Adam/dense_8/kernel/m
 :?2Adam/dense_8/bias/m
&:$	?2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
/:-
??
2Adam/embedding_8/embeddings/v
/:-
??
2Adam/embedding_9/embeddings/v
&:$	?2Adam/dense_8/kernel/v
 :?2Adam/dense_8/bias/v
&:$	?2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
?2?
E__inference_model_4_layer_call_and_return_conditional_losses_10982009
E__inference_model_4_layer_call_and_return_conditional_losses_10982046
E__inference_model_4_layer_call_and_return_conditional_losses_10981922
E__inference_model_4_layer_call_and_return_conditional_losses_10981946?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_model_4_layer_call_fn_10981765
*__inference_model_4_layer_call_fn_10982064
*__inference_model_4_layer_call_fn_10982082
*__inference_model_4_layer_call_fn_10981898?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_10981653?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *O?L
J?G
!?
input_9?????????
"?
input_10?????????
?2?
I__inference_embedding_8_layer_call_and_return_conditional_losses_10982092?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_embedding_8_layer_call_fn_10982099?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_embedding_9_layer_call_and_return_conditional_losses_10982109?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_embedding_9_layer_call_fn_10982116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_8_layer_call_and_return_conditional_losses_10982122?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_8_layer_call_fn_10982127?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_9_layer_call_and_return_conditional_losses_10982133?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_9_layer_call_fn_10982138?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_concatenate_4_layer_call_and_return_conditional_losses_10982145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_concatenate_4_layer_call_fn_10982151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_8_layer_call_and_return_conditional_losses_10982162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_8_layer_call_fn_10982171?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_9_layer_call_and_return_conditional_losses_10982182?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_9_layer_call_fn_10982191?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_10981972input_10input_9"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_10981653?()./Y?V
O?L
J?G
!?
input_9?????????
"?
input_10?????????
? "1?.
,
dense_9!?
dense_9??????????
K__inference_concatenate_4_layer_call_and_return_conditional_losses_10982145?Z?W
P?M
K?H
"?
inputs/0?????????

"?
inputs/1?????????

? "%?"
?
0?????????
? ?
0__inference_concatenate_4_layer_call_fn_10982151vZ?W
P?M
K?H
"?
inputs/0?????????

"?
inputs/1?????????

? "???????????
E__inference_dense_8_layer_call_and_return_conditional_losses_10982162]()/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ~
*__inference_dense_8_layer_call_fn_10982171P()/?,
%?"
 ?
inputs?????????
? "????????????
E__inference_dense_9_layer_call_and_return_conditional_losses_10982182]./0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_dense_9_layer_call_fn_10982191P./0?-
&?#
!?
inputs??????????
? "???????????
I__inference_embedding_8_layer_call_and_return_conditional_losses_10982092_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????

? ?
.__inference_embedding_8_layer_call_fn_10982099R/?,
%?"
 ?
inputs?????????
? "??????????
?
I__inference_embedding_9_layer_call_and_return_conditional_losses_10982109_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????

? ?
.__inference_embedding_9_layer_call_fn_10982116R/?,
%?"
 ?
inputs?????????
? "??????????
?
G__inference_flatten_8_layer_call_and_return_conditional_losses_10982122\3?0
)?&
$?!
inputs?????????

? "%?"
?
0?????????

? 
,__inference_flatten_8_layer_call_fn_10982127O3?0
)?&
$?!
inputs?????????

? "??????????
?
G__inference_flatten_9_layer_call_and_return_conditional_losses_10982133\3?0
)?&
$?!
inputs?????????

? "%?"
?
0?????????

? 
,__inference_flatten_9_layer_call_fn_10982138O3?0
)?&
$?!
inputs?????????

? "??????????
?
E__inference_model_4_layer_call_and_return_conditional_losses_10981922?()./a?^
W?T
J?G
!?
input_9?????????
"?
input_10?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_4_layer_call_and_return_conditional_losses_10981946?()./a?^
W?T
J?G
!?
input_9?????????
"?
input_10?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_model_4_layer_call_and_return_conditional_losses_10982009?()./b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_4_layer_call_and_return_conditional_losses_10982046?()./b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_model_4_layer_call_fn_10981765?()./a?^
W?T
J?G
!?
input_9?????????
"?
input_10?????????
p 

 
? "???????????
*__inference_model_4_layer_call_fn_10981898?()./a?^
W?T
J?G
!?
input_9?????????
"?
input_10?????????
p

 
? "???????????
*__inference_model_4_layer_call_fn_10982064?()./b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
*__inference_model_4_layer_call_fn_10982082?()./b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
&__inference_signature_wrapper_10981972?()./k?h
? 
a?^
.
input_10"?
input_10?????????
,
input_9!?
input_9?????????"1?.
,
dense_9!?
dense_9?????????