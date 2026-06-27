

**unetpp scratch:**

def conv\_block(x, filters, dropout=0.0):

&#x20;   x = layers.Conv2D(filters, 3, padding="same")(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.Activation("relu")(x)

&#x20;   if dropout > 0:

&#x20;       x = layers.Dropout(dropout)(x)

&#x20;   x = layers.Conv2D(filters, 3, padding="same")(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.Activation("relu")(x)

&#x20;   return x



def UNetPlusPlus\_Balanced(input\_shape, num\_classes):

&#x20;   inputs = layers.Input(input\_shape)

&#x20;   filters = \[64, 128, 256, 512, 1024]

&#x20;   

&#x20;   x0\_0 = conv\_block(inputs, filters\[0])

&#x20;   x1\_0 = conv\_block(layers.MaxPooling2D()(x0\_0), filters\[1])

&#x20;   x2\_0 = conv\_block(layers.MaxPooling2D()(x1\_0), filters\[2])

&#x20;   x3\_0 = conv\_block(layers.MaxPooling2D()(x2\_0), filters\[3])

&#x20;   

&#x20;   x0\_1 = conv\_block(layers.concatenate(\[x0\_0, layers.UpSampling2D()(x1\_0)]), filters\[0])

&#x20;   x1\_1 = conv\_block(layers.concatenate(\[x1\_0, layers.UpSampling2D()(x2\_0)]), filters\[1])

&#x20;   x2\_1 = conv\_block(layers.concatenate(\[x2\_0, layers.UpSampling2D()(x3\_0)]), filters\[2])

&#x20;   

&#x20;   x0\_2 = conv\_block(layers.concatenate(\[x0\_0, x0\_1, layers.UpSampling2D()(x1\_1)]), filters\[0])

&#x20;   x1\_2 = conv\_block(layers.concatenate(\[x1\_0, x1\_1, layers.UpSampling2D()(x2\_1)]), filters\[1])

&#x20;   

&#x20;   x0\_3 = conv\_block(layers.concatenate(\[x0\_0, x0\_1, x0\_2, layers.UpSampling2D()(x1\_2)]), filters\[0])

&#x20;   x0\_3 = layers.Dropout(0.1)(x0\_3)

&#x20;   outputs = layers.Conv2D(num\_classes, 1, activation="softmax")(x0\_3)

&#x20;   return models.Model(inputs, outputs)





**unet scratch:**

def conv\_block(x, filters, dropout=0.0):

&#x20;   x = layers.Conv2D(filters, 3, padding="same")(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.Activation("relu")(x)

&#x20;   if dropout > 0:

&#x20;       x = layers.Dropout(dropout)(x)

&#x20;   x = layers.Conv2D(filters, 3, padding="same")(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.Activation("relu")(x)

&#x20;   return x



def UNetPlusPlus\_Balanced(input\_shape, num\_classes):

&#x20;   inputs = layers.Input(input\_shape)

&#x20;   enc\_filters = \[64, 128, 256, 512]

&#x20;   skips = \[]

&#x20;   x = inputs

&#x20;   for f in enc\_filters:

&#x20;       x = conv\_block(x, f)

&#x20;       skips.append(x)

&#x20;       x = layers.MaxPooling2D()(x)

&#x20;   x = conv\_block(x, 1024)               # bottleneck

&#x20;   for i, f in enumerate(reversed(enc\_filters)):

&#x20;       x = layers.UpSampling2D()(x)

&#x20;       x = layers.concatenate(\[x, skips\[-(i+1)]])

&#x20;       x = conv\_block(x, f)

&#x20;   outputs = layers.Conv2D(num\_classes, 1, activation="softmax")(x)

&#x20;   return models.Model(inputs, outputs)







**segformer scratch:**

class OverlapPatchEmbed(layers.Layer):

&#x20;   def \_\_init\_\_(self, patch\_size, stride, embed\_dim, \*\*kwargs):

&#x20;       super().\_\_init\_\_(\*\*kwargs)

&#x20;       self.patch\_size = patch\_size

&#x20;       self.stride = stride

&#x20;       self.embed\_dim = embed\_dim

&#x20;       self.proj = layers.Conv2D(embed\_dim, kernel\_size=patch\_size, strides=stride, padding='same')

&#x20;       self.norm = layers.LayerNormalization(epsilon=1e-6)

&#x20;   def call(self, x):

&#x20;       x = self.proj(x)

&#x20;       H, W = tf.shape(x)\[1], tf.shape(x)\[2]

&#x20;       x = tf.reshape(x, \[tf.shape(x)\[0], H \* W, self.embed\_dim])

&#x20;       x = self.norm(x)

&#x20;       return x, H, W

&#x20;   def get\_config(self):

&#x20;       cfg = super().get\_config()

&#x20;       cfg.update({'patch\_size': self.patch\_size, 'stride': self.stride, 'embed\_dim': self.embed\_dim})

&#x20;       return cfg



class EfficientAttention(layers.Layer):

&#x20;   def \_\_init\_\_(self, dim, num\_heads, sr\_ratio, \*\*kwargs):

&#x20;       super().\_\_init\_\_(\*\*kwargs)

&#x20;       self.dim = dim

&#x20;       self.num\_heads = num\_heads

&#x20;       self.sr\_ratio = sr\_ratio

&#x20;       self.scale = (dim // num\_heads) \*\* -0.5

&#x20;       self.q = layers.Dense(dim)

&#x20;       self.kv = layers.Dense(dim \* 2)

&#x20;       self.proj = layers.Dense(dim)

&#x20;       if sr\_ratio > 1:

&#x20;           self.sr = layers.Conv2D(dim, kernel\_size=sr\_ratio, strides=sr\_ratio, padding='same')

&#x20;           self.norm = layers.LayerNormalization(epsilon=1e-6)

&#x20;   def call(self, x, H, W):

&#x20;       B = tf.shape(x)\[0]

&#x20;       N = tf.shape(x)\[1]

&#x20;       q = self.q(x)

&#x20;       q = tf.reshape(q, \[B, N, self.num\_heads, self.dim // self.num\_heads])

&#x20;       q = tf.transpose(q, \[0, 2, 1, 3])

&#x20;       if self.sr\_ratio > 1:

&#x20;           x\_ = tf.reshape(x, \[B, H, W, self.dim])

&#x20;           x\_ = self.sr(x\_)

&#x20;           x\_ = tf.reshape(x\_, \[B, -1, self.dim])

&#x20;           x\_ = self.norm(x\_)

&#x20;           kv = self.kv(x\_)

&#x20;       else:

&#x20;           kv = self.kv(x)

&#x20;       kv = tf.reshape(kv, \[B, -1, 2, self.num\_heads, self.dim // self.num\_heads])

&#x20;       kv = tf.transpose(kv, \[2, 0, 3, 1, 4])

&#x20;       k, v = kv\[0], kv\[1]

&#x20;       attn = tf.matmul(q, k, transpose\_b=True) \* self.scale

&#x20;       attn = tf.nn.softmax(attn, axis=-1)

&#x20;       x = tf.matmul(attn, v)

&#x20;       x = tf.transpose(x, \[0, 2, 1, 3])

&#x20;       x = tf.reshape(x, \[B, N, self.dim])

&#x20;       x = self.proj(x)

&#x20;       return x

&#x20;   def get\_config(self):

&#x20;       cfg = super().get\_config()

&#x20;       cfg.update({'dim': self.dim, 'num\_heads': self.num\_heads, 'sr\_ratio': self.sr\_ratio})

&#x20;       return cfg



class TransformerBlock(layers.Layer):

&#x20;   def \_\_init\_\_(self, dim, num\_heads, mlp\_ratio, sr\_ratio, drop=0., \*\*kwargs):

&#x20;       super().\_\_init\_\_(\*\*kwargs)

&#x20;       self.dim = dim

&#x20;       self.num\_heads = num\_heads

&#x20;       self.mlp\_ratio = mlp\_ratio

&#x20;       self.sr\_ratio = sr\_ratio

&#x20;       self.drop = drop

&#x20;       self.norm1 = layers.LayerNormalization(epsilon=1e-6)

&#x20;       self.attn = EfficientAttention(dim, num\_heads, sr\_ratio)

&#x20;       self.norm2 = layers.LayerNormalization(epsilon=1e-6)

&#x20;       mlp\_hidden\_dim = int(dim \* mlp\_ratio)

&#x20;       self.mlp = tf.keras.Sequential(\[

&#x20;           layers.Dense(mlp\_hidden\_dim, activation=tf.nn.gelu),

&#x20;           layers.Dropout(drop),

&#x20;           layers.Dense(dim),

&#x20;           layers.Dropout(drop)

&#x20;       ])

&#x20;   def call(self, x, H, W):

&#x20;       x = x + self.attn(self.norm1(x), H, W)

&#x20;       x = x + self.mlp(self.norm2(x))

&#x20;       return x

&#x20;   def get\_config(self):

&#x20;       cfg = super().get\_config()

&#x20;       cfg.update({

&#x20;           'dim': self.dim, 'num\_heads': self.num\_heads,

&#x20;           'mlp\_ratio': self.mlp\_ratio, 'sr\_ratio': self.sr\_ratio, 'drop': self.drop

&#x20;       })

&#x20;       return cfg



class MixTransformerStage(layers.Layer):

&#x20;   def \_\_init\_\_(self, embed\_dim, num\_heads, mlp\_ratio, sr\_ratio, depth, patch\_size, stride, \*\*kwargs):

&#x20;       super().\_\_init\_\_(\*\*kwargs)

&#x20;       self.embed\_dim = embed\_dim

&#x20;       self.num\_heads = num\_heads

&#x20;       self.mlp\_ratio = mlp\_ratio

&#x20;       self.sr\_ratio = sr\_ratio

&#x20;       self.depth = depth

&#x20;       self.patch\_size = patch\_size

&#x20;       self.stride = stride

&#x20;       self.patch\_embed = OverlapPatchEmbed(patch\_size, stride, embed\_dim)

&#x20;       self.blocks = \[TransformerBlock(embed\_dim, num\_heads, mlp\_ratio, sr\_ratio) for \_ in range(depth)]

&#x20;       self.norm = layers.LayerNormalization(epsilon=1e-6)

&#x20;   def call(self, x):

&#x20;       x, H, W = self.patch\_embed(x)

&#x20;       for block in self.blocks:

&#x20;           x = block(x, H, W)

&#x20;       x = self.norm(x)

&#x20;       x = tf.reshape(x, \[tf.shape(x)\[0], H, W, self.embed\_dim])

&#x20;       return x

&#x20;   def get\_config(self):

&#x20;       cfg = super().get\_config()

&#x20;       cfg.update({

&#x20;           'embed\_dim': self.embed\_dim, 'num\_heads': self.num\_heads,

&#x20;           'mlp\_ratio': self.mlp\_ratio, 'sr\_ratio': self.sr\_ratio,

&#x20;           'depth': self.depth, 'patch\_size': self.patch\_size, 'stride': self.stride

&#x20;       })

&#x20;       return cfg



class MixTransformerEncoder(layers.Layer):

&#x20;   def \_\_init\_\_(self, embed\_dims, num\_heads, mlp\_ratios, sr\_ratios, depths, \*\*kwargs):

&#x20;       super().\_\_init\_\_(\*\*kwargs)

&#x20;       self.embed\_dims = embed\_dims

&#x20;       self.num\_heads = num\_heads

&#x20;       self.mlp\_ratios = mlp\_ratios

&#x20;       self.sr\_ratios = sr\_ratios

&#x20;       self.depths = depths

&#x20;       self.stages = \[]

&#x20;       for i in range(len(depths)):

&#x20;           patch\_size = 7 if i == 0 else 3

&#x20;           stride = 4 if i == 0 else 2

&#x20;           stage = MixTransformerStage(

&#x20;               embed\_dims\[i], num\_heads\[i], mlp\_ratios\[i], sr\_ratios\[i],

&#x20;               depths\[i], patch\_size, stride, name=f'stage\_{i}'

&#x20;           )

&#x20;           self.stages.append(stage)

&#x20;   def call(self, x):

&#x20;       outputs = \[]

&#x20;       for stage in self.stages:

&#x20;           x = stage(x)

&#x20;           outputs.append(x)

&#x20;       return outputs

&#x20;   def get\_config(self):

&#x20;       cfg = super().get\_config()

&#x20;       cfg.update({

&#x20;           'embed\_dims': self.embed\_dims, 'num\_heads': self.num\_heads,

&#x20;           'mlp\_ratios': self.mlp\_ratios, 'sr\_ratios': self.sr\_ratios, 'depths': self.depths

&#x20;       })

&#x20;       return cfg



class MLPDecoder(layers.Layer):

&#x20;   def \_\_init\_\_(self, num\_classes, embed\_dims, \*\*kwargs):

&#x20;       super().\_\_init\_\_(\*\*kwargs)

&#x20;       self.num\_classes = num\_classes

&#x20;       self.embed\_dims = embed\_dims

&#x20;       self.linear\_c4 = layers.Conv2D(embed\_dims\[0], 1, name='linear\_c4')

&#x20;       self.linear\_c3 = layers.Conv2D(embed\_dims\[0], 1, name='linear\_c3')

&#x20;       self.linear\_c2 = layers.Conv2D(embed\_dims\[0], 1, name='linear\_c2')

&#x20;       self.linear\_c1 = layers.Conv2D(embed\_dims\[0], 1, name='linear\_c1')

&#x20;       self.linear\_fuse = tf.keras.Sequential(\[

&#x20;           layers.Conv2D(embed\_dims\[0], 1, use\_bias=False, name='fuse\_conv'),

&#x20;           layers.BatchNormalization(name='fuse\_bn'),

&#x20;           layers.Activation('relu'),

&#x20;           layers.Dropout(0.1, name='fuse\_drop')

&#x20;       ], name='linear\_fuse')

&#x20;       self.linear\_pred = layers.Conv2D(num\_classes, 1, name='linear\_pred')

&#x20;   def call(self, inputs):

&#x20;       c1, c2, c3, c4 = inputs

&#x20;       n, m = tf.shape(c1)\[1], tf.shape(c1)\[2]

&#x20;       \_c4 = tf.image.resize(self.linear\_c4(c4), \[n, m], method='bilinear')

&#x20;       \_c3 = tf.image.resize(self.linear\_c3(c3), \[n, m], method='bilinear')

&#x20;       \_c2 = tf.image.resize(self.linear\_c2(c2), \[n, m], method='bilinear')

&#x20;       \_c1 = self.linear\_c1(c1)

&#x20;       \_c = tf.concat(\[\_c1, \_c2, \_c3, \_c4], axis=-1)

&#x20;       \_c = self.linear\_fuse(\_c)

&#x20;       return self.linear\_pred(\_c)

&#x20;   def get\_config(self):

&#x20;       cfg = super().get\_config()

&#x20;       cfg.update({'num\_classes': self.num\_classes, 'embed\_dims': self.embed\_dims})

&#x20;       return cfg



class ResizeLayer(layers.Layer):

&#x20;   def \_\_init\_\_(self, method='bilinear', \*\*kwargs):

&#x20;       super().\_\_init\_\_(\*\*kwargs)

&#x20;       self.method = method

&#x20;   def call(self, inputs):

&#x20;       x, reference = inputs

&#x20;       return tf.image.resize(x, \[tf.shape(reference)\[1], tf.shape(reference)\[2]], method=self.method)

&#x20;   def get\_config(self):

&#x20;       cfg = super().get\_config()

&#x20;       cfg.update({'method': self.method})

&#x20;       return cfg



def SegFormer(input\_shape, num\_classes,

&#x20;             embed\_dims=\[32, 64, 160, 256],

&#x20;             num\_heads=\[1, 2, 5, 8],

&#x20;             mlp\_ratios=\[4, 4, 4, 4],

&#x20;             sr\_ratios=\[8, 4, 2, 1],

&#x20;             depths=\[2, 2, 2, 2],

&#x20;             name='SegFormer'):

&#x20;   inputs = layers.Input(input\_shape, name='input')

&#x20;   encoder = MixTransformerEncoder(

&#x20;       embed\_dims=embed\_dims, num\_heads=num\_heads,

&#x20;       mlp\_ratios=mlp\_ratios, sr\_ratios=sr\_ratios, depths=depths,

&#x20;       name='encoder'

&#x20;   )

&#x20;   features = encoder(inputs)

&#x20;   decoder = MLPDecoder(num\_classes=num\_classes, embed\_dims=embed\_dims, name='decoder')

&#x20;   x = decoder(features)

&#x20;   x = ResizeLayer(method='bilinear')(\[x, inputs])

&#x20;   outputs = layers.Activation('softmax', name='softmax')(x)

&#x20;   return models.Model(inputs, outputs, name=name)





**deeplab scratch:**

def DeepLabV3Plus\_scratch(input\_shape, num\_classes, output\_stride=16):

&#x20;   inputs = layers.Input(input\_shape)

&#x20;   h, w, c = input\_shape  # (256, 256, 10)



&#x20;   # ResNet101 from scratch, N-band input (no pretrained weights)

&#x20;   base\_model = ResNet101(

&#x20;       include\_top=False,

&#x20;       weights=None,               # train from scratch

&#x20;       input\_shape=(h, w, c),

&#x20;       pooling=None

&#x20;   )



&#x20;   # Feature extraction layers

&#x20;   low\_level\_layer = 'conv2\_block3\_out'       # (64, 64, 256)

&#x20;   high\_level\_layer = 'conv4\_block23\_out'     # (16, 16, 1024) – ResNet101 conv4 has 23 blocks



&#x20;   intermediate\_model = models.Model(

&#x20;       inputs=base\_model.input,

&#x20;       outputs=\[base\_model.get\_layer(low\_level\_layer).output,

&#x20;                base\_model.get\_layer(high\_level\_layer).output]

&#x20;   )



&#x20;   low\_features, high\_features = intermediate\_model(inputs)



&#x20;   # ----- ASPP (output\_stride=16 → rates \[6,12,18]) -----

&#x20;   rates = \[6, 12, 18]



&#x20;   b0 = layers.Conv2D(256, 1, padding='same', use\_bias=False)(high\_features)

&#x20;   b0 = layers.BatchNormalization()(b0)

&#x20;   b0 = layers.ReLU()(b0)



&#x20;   branches = \[b0]

&#x20;   for rate in rates:

&#x20;       b = layers.Conv2D(256, 3, dilation\_rate=rate, padding='same', use\_bias=False)(high\_features)

&#x20;       b = layers.BatchNormalization()(b)

&#x20;       b = layers.ReLU()(b)

&#x20;       branches.append(b)



&#x20;   # Image‑level features

&#x20;   b\_img = layers.GlobalAveragePooling2D()(high\_features)

&#x20;   b\_img = layers.Reshape((1, 1, 1024))(b\_img)

&#x20;   b\_img = layers.Conv2D(256, 1, padding='same', use\_bias=False)(b\_img)

&#x20;   b\_img = layers.BatchNormalization()(b\_img)

&#x20;   b\_img = layers.ReLU()(b\_img)

&#x20;   b\_img = layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(b\_img)

&#x20;   branches.append(b\_img)



&#x20;   x\_aspp = layers.Concatenate()(branches)

&#x20;   x\_aspp = layers.Conv2D(256, 1, padding='same', use\_bias=False)(x\_aspp)

&#x20;   x\_aspp = layers.BatchNormalization()(x\_aspp)

&#x20;   x\_aspp = layers.ReLU()(x\_aspp)



&#x20;   # ----- Decoder -----

&#x20;   x\_decoder = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x\_aspp)  # → (64,64)



&#x20;   low\_features = layers.Conv2D(48, 1, padding='same', use\_bias=False)(low\_features)

&#x20;   low\_features = layers.BatchNormalization()(low\_features)

&#x20;   low\_features = layers.ReLU()(low\_features)



&#x20;   x = layers.Concatenate()(\[x\_decoder, low\_features])   # 256 + 48 = 304

&#x20;   x = layers.Conv2D(256, 3, padding='same', use\_bias=False)(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.ReLU()(x)

&#x20;   x = layers.Conv2D(256, 3, padding='same', use\_bias=False)(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.ReLU()(x)



&#x20;   x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)   # → (256,256)



&#x20;   outputs = layers.Conv2D(num\_classes, 1, activation='softmax')(x)



&#x20;   model = models.Model(inputs, outputs, name='DeepLabV3Plus\_ResNet101\_scratch')

&#x20;   return model





**bilstm scratch:**

def CNN\_BiLSTM\_SegNet(input\_shape, num\_classes):

&#x20;      inputs = layers.Input(input\_shape)

&#x20;   h, w, actual\_channels = input\_shape\[0], input\_shape\[1], input\_shape\[2]



&#x20;   from tensorflow.keras.applications import ResNet101

&#x20;   base\_model = ResNet101(

&#x20;       include\_top=False,

&#x20;       weights=None,               # train from scratch

&#x20;       input\_shape=(h, w, actual\_channels),

&#x20;       pooling=None

&#x20;   )



&#x20;   low\_level\_layer = 'conv2\_block3\_out'

&#x20;   skip2\_layer      = 'conv3\_block4\_out'

&#x20;   skip3\_layer      = 'conv4\_block23\_out'



&#x20;   rest\_model = models.Model(

&#x20;       inputs=base\_model.input,

&#x20;       outputs=\[

&#x20;           base\_model.get\_layer(low\_level\_layer).output,

&#x20;           base\_model.get\_layer(skip2\_layer).output,

&#x20;           base\_model.get\_layer(skip3\_layer).output,

&#x20;       ]

&#x20;   )

&#x20;   low\_feat, skip2\_feat, high\_feat = rest\_model(inputs)



&#x20;   hw, ww, c = 16, 16, 1024  # for 256×256 input



&#x20;   # Row-wise BiLSTM

&#x20;   x = layers.Reshape((hw, ww \* c))(high\_feat)

&#x20;   x = layers.Bidirectional(layers.LSTM(256, return\_sequences=True))(x)

&#x20;   x = layers.Bidirectional(layers.LSTM(256, return\_sequences=True))(x)

&#x20;   x = layers.Dense(ww \* 256)(x)

&#x20;   row\_out = layers.Reshape((hw, ww, 256))(x)



&#x20;   # Column-wise BiLSTM

&#x20;   x = layers.Permute((2, 1, 3))(row\_out)

&#x20;   x = layers.Reshape((ww, hw \* 256))(x)

&#x20;   x = layers.Bidirectional(layers.LSTM(256, return\_sequences=True))(x)

&#x20;   x = layers.Bidirectional(layers.LSTM(256, return\_sequences=True))(x)

&#x20;   x = layers.Dense(hw \* 256)(x)

&#x20;   x = layers.Reshape((ww, hw, 256))(x)

&#x20;   col\_out = layers.Permute((2, 1, 3))(x)



&#x20;   x = layers.Concatenate()(\[row\_out, col\_out])

&#x20;   x = layers.Conv2D(256, 1, padding='same')(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.ReLU()(x)



&#x20;   # Decoder

&#x20;   x = layers.UpSampling2D(2, interpolation='bilinear')(x)

&#x20;   skip2\_proj = layers.Conv2D(128, 1, padding='same')(skip2\_feat)

&#x20;   skip2\_proj = layers.BatchNormalization()(skip2\_proj)

&#x20;   skip2\_proj = layers.ReLU()(skip2\_proj)

&#x20;   x = layers.Concatenate()(\[x, skip2\_proj])

&#x20;   x = layers.Conv2D(128, 3, padding='same')(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.ReLU()(x)



&#x20;   x = layers.UpSampling2D(2, interpolation='bilinear')(x)

&#x20;   low\_proj = layers.Conv2D(64, 1, padding='same')(low\_feat)

&#x20;   low\_proj = layers.BatchNormalization()(low\_proj)

&#x20;   low\_proj = layers.ReLU()(low\_proj)

&#x20;   x = layers.Concatenate()(\[x, low\_proj])

&#x20;   x = layers.Conv2D(64, 3, padding='same')(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.ReLU()(x)



&#x20;   x = layers.UpSampling2D(2, interpolation='bilinear')(x)

&#x20;   x = layers.Conv2D(32, 3, padding='same')(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.ReLU()(x)



&#x20;   x = layers.UpSampling2D(2, interpolation='bilinear')(x)

&#x20;   x = layers.Conv2D(32, 3, padding='same')(x)

&#x20;   x = layers.BatchNormalization()(x)

&#x20;   x = layers.ReLU()(x)



&#x20;   outputs = layers.Conv2D(num\_classes, 1, activation='softmax')(x)



&#x20;   model = models.Model(inputs, outputs, name='CNN\_BiLSTM\_SegNet')

&#x20;   return model