//#region Math
export function getIdentity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
}

export function times(left: number[], right: number[]): number[] {
  return [
    left[0x0] * right[0x0] + left[0x4] * right[0x1] + left[0x8] * right[0x2] + left[0xc] * right[0x3],
    left[0x1] * right[0x0] + left[0x5] * right[0x1] + left[0x9] * right[0x2] + left[0xd] * right[0x3],
    left[0x2] * right[0x0] + left[0x6] * right[0x1] + left[0xa] * right[0x2] + left[0xe] * right[0x3],
    left[0x3] * right[0x0] + left[0x7] * right[0x1] + left[0xb] * right[0x2] + left[0xf] * right[0x3],
    left[0x0] * right[0x4] + left[0x4] * right[0x5] + left[0x8] * right[0x6] + left[0xc] * right[0x7],
    left[0x1] * right[0x4] + left[0x5] * right[0x5] + left[0x9] * right[0x6] + left[0xd] * right[0x7],
    left[0x2] * right[0x4] + left[0x6] * right[0x5] + left[0xa] * right[0x6] + left[0xe] * right[0x7],
    left[0x3] * right[0x4] + left[0x7] * right[0x5] + left[0xb] * right[0x6] + left[0xf] * right[0x7],
    left[0x0] * right[0x8] + left[0x4] * right[0x9] + left[0x8] * right[0xa] + left[0xc] * right[0xb],
    left[0x1] * right[0x8] + left[0x5] * right[0x9] + left[0x9] * right[0xa] + left[0xd] * right[0xb],
    left[0x2] * right[0x8] + left[0x6] * right[0x9] + left[0xa] * right[0xa] + left[0xe] * right[0xb],
    left[0x3] * right[0x8] + left[0x7] * right[0x9] + left[0xb] * right[0xa] + left[0xf] * right[0xb],
    left[0x0] * right[0xc] + left[0x4] * right[0xd] + left[0x8] * right[0xe] + left[0xc] * right[0xf],
    left[0x1] * right[0xc] + left[0x5] * right[0xd] + left[0x9] * right[0xe] + left[0xd] * right[0xf],
    left[0x2] * right[0xc] + left[0x6] * right[0xd] + left[0xa] * right[0xe] + left[0xe] * right[0xf],
    left[0x3] * right[0xc] + left[0x7] * right[0xd] + left[0xb] * right[0xe] + left[0xf] * right[0xf],
  ]
}

export function invert(A: number[]) {
  const n = 4
  const m = 8
  const augmented: Array<number>[] = []
  for (let i = 0; i < n; ++i) {
    augmented[i] = new Array(m).fill(0)
    for (let j = 0; j < n; ++j) {
      augmented[i][j] = A[j * n + i]
    }
    augmented[i][i + n] = 1.0
  }
  for (let i = 0; i < n; ++i) {
    let pivot = i
    for (let j = i + 1; j < n; ++j) {
      if (Math.abs(augmented[j][i]) > Math.abs(augmented[pivot][i])) {
        pivot = j
      }
    }
    if (Math.abs(augmented[pivot][i]) < Number.EPSILON) continue
    if (pivot != i) {
      for (let j = i; j < m; ++j) {
        const t = augmented[i][j]
        augmented[i][j] = augmented[pivot][j]
        augmented[pivot][j] = t
      }
    }
    for (let j = i + 1; j < m; ++j) {
      augmented[i][j] /= augmented[i][i]
    }
    for (let j = 0; j < n; ++j) {
      if (j == i) continue
      for (let k = i + 1; k < m; ++k) {
        augmented[j][k] -= augmented[i][k] * augmented[j][i]
      }
    }
  }
  return [
    augmented[0][4], augmented[1][4], augmented[2][4], augmented[3][4],
    augmented[0][5], augmented[1][5], augmented[2][5], augmented[3][5],
    augmented[0][6], augmented[1][6], augmented[2][6], augmented[3][6],
    augmented[0][7], augmented[1][7], augmented[2][7], augmented[3][7]
  ]
}

export function composeTransformation(scale: number[], rotation: number[], translation: number[]) {
  const x2 = rotation[0] + rotation[0]
  const y2 = rotation[1] + rotation[1]
  const z2 = rotation[2] + rotation[2]

  const xx2 = x2 * rotation[0]
  const xy2 = x2 * rotation[1]
  const xz2 = x2 * rotation[2]
  const xw2 = x2 * rotation[3]

  const yy2 = y2 * rotation[1]
  const yz2 = y2 * rotation[2]
  const yw2 = y2 * rotation[3]

  const zz2 = z2 * rotation[2]
  const zw2 = z2 * rotation[3]

  return [
    (1 - yy2 - zz2) * scale[0], (xy2 + zw2) * scale[0], (xz2 - yw2) * scale[0], 0,
    (xy2 - zw2) * scale[1], (1 - xx2 - zz2) * scale[1], (yz2 + xw2) * scale[1], 0,
    (xz2 + yw2) * scale[2], (yz2 - xw2) * scale[2], (1 - xx2 - yy2) * scale[2], 0,
    translation[0], translation[1], translation[2], 1
  ]
}

export function normalize(a: number[]): number[] {
  const len = Math.sqrt(a.reduce((sum, i) => sum + i * i, 0))
  return a.map(i => i / len)
}

export function cross(a: number[], b: number[]): number[] {
  return [ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] ]
}

export function dot(a: number[], b: number[]) {
  return a.reduce((sum, i, j) => sum + i * b[j], 0)
}

export function perspective(fov: number, near: number, far: number, aspect: number) {
  const f = 1.0 / Math.tan(fov * Math.PI / 360.0)
  const depthInv = 1.0 / (near - far)

  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (near + far) * depthInv, -1,
    0, 0, near * far * depthInv * 2, 0
  ]
}

export function lookAt(position: number[], target: number[], up: number[]) {
  const gaze = normalize([ position[0] - target[0], position[1] - target[1], position[2] - target[2]])

  const right = normalize(cross(up, gaze))
  up = cross(gaze, right)

  return [
    right[0], up[0], gaze[0], 0,
    right[1], up[1], gaze[1], 0,
    right[2], up[2], gaze[2], 0,
    -dot(right, position), -dot(up, position), -dot(gaze, position), 1
  ]
}

export function lerp(previous: number[], next: number[], interpolation: number) {
  return next.map((value, index) => {
    return previous[index] + interpolation * (value - previous[index])
  })
}

export function slerp(previous: number[], next: number[], interpolation: number) {
  let d = dot(previous, next)
  if (d < 0.0) {
    next = next.map(n => { return -n })
    d = -d
  }
  if (d > 0.9995) {
    return normalize(lerp(previous, next, interpolation))
  }
  const theta_0 = Math.acos(d)
  const sin_theta_0 = Math.sin(theta_0)
  const theta = interpolation * theta_0
  const sin_theta = Math.sin(theta)
  const scaleNextQuat = sin_theta / sin_theta_0
  const scalePreviousQuat = Math.cos(theta) - d * scaleNextQuat
  return next.map((value, index) => {
    return scalePreviousQuat * previous[index] + scaleNextQuat * value
  })
}
//#endregion

//#region shader
export interface ShaderSourceProvider {
  getVertexSource(): string 
  getFragementSource(): string
}

export class TextureShaderSourceProvider implements ShaderSourceProvider {
  getVertexSource() {
    return `#version 300 es
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec2 texCoord;
    layout(location = 2) in vec3 normal;
    uniform mat4 camera;
    uniform mat4 model;
    uniform mat4 mesh;
    out vec2 uv;
    void main() {
      gl_Position = camera * model * mesh * vec4(position, 1.0);
      uv = texCoord;
    }`
  }

  getFragementSource() {
    return `#version 300 es
    precision highp float;
    in vec2 uv;
    uniform sampler2D sampler;
    out vec4 FragColor;
    void main() {
      FragColor = texture(sampler, uv);
    }`
  }
}

export class SkinningMeshShaderSourceProvider implements ShaderSourceProvider {
  getVertexSource() {
    return `#version 300 es
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec2 texCoord;
    layout(location = 2) in vec3 normal;
    layout(location = 3) in vec4 joint;
    layout(location = 4) in vec4 weight;
    uniform mat4 camera;
    uniform mat4 model;
    uniform mat4 mesh;
    uniform mat4 joints[64];
    out vec2 uv;
    void main() {
      mat4 skin =
        weight.x * joints[int(joint.x)] +
        weight.y * joints[int(joint.y)] +
        weight.z * joints[int(joint.z)] +
        weight.w * joints[int(joint.w)];
      gl_Position = camera * model * skin * mesh * vec4(position, 1.0);
      uv = texCoord / 4096.0;
    }`
  }

  getFragementSource() {
    return `#version 300 es
    precision highp float;
    in vec2 uv;
    uniform sampler2D sampler;
    out vec4 FragColor;
    void main() {
      FragColor = texture(sampler, uv);
      //FragColor = vec4(0.5, 0.5, 0.5, 1.0);
    }`
  }
}

export class Shader {
  constructor(readonly gl: WebGL2RenderingContext, shaderSourceProvider: ShaderSourceProvider) {
    const vs = gl.createShader(gl.VERTEX_SHADER)
    const fs = gl.createShader(gl.FRAGMENT_SHADER)
    if (vs === null || fs === null) {
      throw new Error("WebGL failed at creating shader")
    }
    gl.shaderSource(vs, shaderSourceProvider.getVertexSource())
    gl.shaderSource(fs, shaderSourceProvider.getFragementSource())
    gl.compileShader(vs)
    gl.compileShader(fs)
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`WebGL failed at compiling vertex shader ${gl.getShaderInfoLog(vs)}`)
    }
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`WebGL failed at compiling fragment shader ${gl.getShaderInfoLog(fs)}`)
    }
    const program = gl.createProgram()
    if (program === null) {
      throw new Error("WebGL failed at creating program")
    }
    gl.attachShader(program, vs)
    gl.attachShader(program, fs)
    gl.linkProgram(program)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`WebGL failed at linking program ${gl.getProgramInfoLog(program)}`)
    }
    this.program = program
  }

  use() {
    this.gl.useProgram(this.program)
  }

  setMatrix(name: string, value: number[]) {
    if (!this.uniforms.has(name)) {
      this.uniforms.set(name, this.gl.getUniformLocation(this.program, name))
    }
    this.gl.uniformMatrix4fv(this.uniforms.get(name), false, value)
  }

  program: WebGLProgram
  uniforms = new Map<string, WebGLUniformLocation>
}
//#endregion

export class GLB {
  constructor(readonly gltf: any, readonly bin: ArrayBuffer) {}

  static async loadFrom(uri: string) {
    const buffer = await fetch(`${uri}`).then(response => response.arrayBuffer())
    const [magic, version, length] = new Uint32Array(buffer.slice(0, 12))
    if (magic != 1179937895 || version != 2) {
      console.log("Invalid glb")
    }
    const [gltfLength, gltfType] = new Uint32Array(buffer.slice(12, 20))
    if (gltfType != 1313821514) {
      console.log("Invalid glTF chunk")
    }
    const gltf = JSON.parse(new TextDecoder("utf-8").decode(buffer.slice(20, 20 + gltfLength)))
    const [binLength, binType] = new Uint32Array(buffer.slice(20 + gltfLength, 28 + gltfLength))
    if (binType != 5130562) {
      console.log("Invalid bin chunk")
    }
    if (gltfLength + binLength + 28 != length) {
      console.log("Invalid length")
    }
    const bin = buffer.slice(28 + gltfLength)
    return new GLB(gltf, bin)
  }

  access(accessorId: number) {
    const accessor = this.gltf.accessors[accessorId]
    const bufferView = this.gltf.bufferViews[accessor.bufferView]
    const elementSize = GLB.ELEMENT_SIZE[accessor.type]
    const typedArray = GLB.COMPONENT_TYPE[accessor.componentType]
    const unitLength = elementSize * typedArray.BYTES_PER_ELEMENT
    const byteStride = bufferView.byteStride || unitLength
    let byteOffset = (bufferView.byteOffset || 0) + (accessor.byteOffset || 0)
    const data = new typedArray(accessor.count * elementSize)
    for (let i = 0; i < data.length; i += elementSize) {
      const value = new typedArray(this.bin, byteOffset, elementSize)
      for (let j = 0; j < elementSize; ++j) {
        data[i + j] = value[j]
      }
      byteOffset += byteStride
    }
    return {
      "count": accessor.count,
      "size": elementSize,
      "type": accessor.componentType,
      "normalized": accessor.normalized ?? false,
      "data": data,
    }
  }

  private static ELEMENT_SIZE = { 'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16 }
  private static COMPONENT_TYPE = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array }
}

export class Model {
  constructor(
    readonly meshes: Mesh[],
    readonly bindPose: Map<string, number[]>,
    readonly textures: WebGLTexture[],
    public matrix: number[]
  ) {}
}

export class AnimateModel extends Model {
  constructor(
    meshes: Mesh[],
    bindPose: Map<string, number[]>,
    textures: WebGLTexture[],
    matrix: number[],
    readonly skins: any,
    readonly nodes: any
  ) {
    super(meshes, bindPose, textures, matrix)
    bindPose.forEach((v, k) => this.inverseBindMatrices.set(k, invert(v)))
  }

  public async loadAnimation(animationName: string, animationUri: string) {
    const glb = await GLB.loadFrom(animationUri)
    if (glb.gltf.animations === undefined) {
      console.log(`Invalid animation uri ${animationUri}`)
    }
    const animation = glb.gltf.animations[0]
    const modelAnimation = new Animation()
    modelAnimation.nodes = glb.gltf.nodes
    modelAnimation.root = glb.gltf.scenes[glb.gltf.scene ?? 0].nodes[0]
    modelAnimation.inputs = new Map<number, number[]>()
    modelAnimation.channels = animation.channels.map(channel => {
      const sampler = animation.samplers[channel.sampler]
      if (!modelAnimation.inputs.has(sampler.input)) {
        const typedInputs = glb.access(sampler.input).data
        const inputs = new Array<number>()
        typedInputs.forEach(input => inputs.push(input))
        modelAnimation.inputs.set(sampler.input, inputs)
      }
      const typedOutputs = glb.access(sampler.output)
      const outputs = new Array<Array<number>>()
      for (let time = 0; time < typedOutputs.count; ++time) {
        const currentTime = time * typedOutputs.size
        const transformation = new Array<number>(typedOutputs.size)
        for (let j = 0; j < typedOutputs.size; ++j) {
          transformation[j] = typedOutputs.data[currentTime + j]
          if (typedOutputs.normalized) {
            transformation[j] = Math.max(transformation[j] / 32767.0, -1.0)
          }
        }
        outputs.push(transformation)
      }
      return {
        target: channel.target.node,
        path: channel.target.path,
        output: outputs,
        inputIndex: sampler.input
      }
    })
    this.animations.set(animationName, modelAnimation)
    this.currentAnimation = animationName
  }

  public animations = new Map<string, Animation>()
  public currentAnimation: string
  public inverseBindMatrices = new Map<string, number[]>()
}

export class Animation {
  public channels: {
    target: number,
    path: "rotation" | "scale" | "translation",
    output: number[][],
    inputIndex: number
  }[] = []
  public inputs: Map<number, number[]>
  public nodes: any
  public root: number
  
  public getPose() {
    const nodes = this.nodes
    function dfs(nodeId: number, parent: number[]) {
      const node = nodes[nodeId]
      const translation = node.translation || [0, 0, 0]
      const rotation = node.rotation || [0, 0, 0, 0]
      const scale = node.scale|| [1, 1, 1]
      const local = composeTransformation(scale, rotation, translation)
      const current = times(parent, local)
      const pose = new Map<string, number[]>([[node.name, current]])
      if (node.children != undefined) {
        node.children.forEach(child => {
          dfs(child, current).forEach((v, k) => pose.set(k, v))
        })
      }
      return pose
    }
    return dfs(this.root, getIdentity())
  }

  public getFramePose(timestamp: number) {
    const interpolations = new Map<number, {next: number, previous: number, rate: number}>()
    this.inputs.forEach((input, index) => {
      const currentFrameTime = input[input.length - 1] != 0
      ? timestamp - Math.floor(timestamp / input[input.length - 1]) * input[input.length - 1]
      : 0
      let previousFrameIndex = 0
      let nextFrameIndex = 0
      while (nextFrameIndex < input.length) {
        if (input[nextFrameIndex] >= currentFrameTime) break
        previousFrameIndex = nextFrameIndex++
      }
      const rate = nextFrameIndex != previousFrameIndex
        ? (currentFrameTime - input[previousFrameIndex]) / (input[nextFrameIndex] - input[previousFrameIndex])
        : 1
      interpolations.set(index, {
        next: nextFrameIndex,
        previous: previousFrameIndex,
        rate: rate
      })
    })
    this.channels.forEach(channel => {
      const interpolation = interpolations.get(channel.inputIndex)
      const previousFrame = channel.output[interpolation.previous]
      const nextFrame = channel.output[interpolation.next]
      const currentFrame = channel.path == "rotation"
        ? slerp(previousFrame, nextFrame, interpolation.rate)
        : lerp(previousFrame, nextFrame, interpolation.rate)
      this.nodes[channel.target][channel.path] = currentFrame
    })
    return this.getPose()
  }
}

export class Mesh {
  constructor(
    public readonly matrix: number[],
    public readonly primitives: Primitive[],
    public readonly skinIndex: number
  ) {}
}

export class Primitive {
  constructor(
    public readonly vao: WebGLVertexArrayObject,
    public readonly count: number,
    public readonly type: number,
    public readonly textureId: number
  ) {}
}

export function getBindPose(glb: GLB) {
  const bindPose = new Map<string, number[]>()
  function dfs(nodeId: number, parentTransformation: number[], parentName?: string) {
    const node = glb.gltf.nodes[nodeId]
    const translation = node.translation || [0, 0, 0]
    const rotation = node.rotation || [0, 0, 0, 0]
    const scale = node.scale|| [1, 1, 1]
    const localTransformation = node.matrix !== undefined ? node.matrix : composeTransformation(scale, rotation, translation)
    const globalTransformation = times(parentTransformation, localTransformation)
    bindPose.set(node.name ?? nodeId, globalTransformation)
    if (node.children !== undefined) {
      node.children.forEach(child => dfs(child, globalTransformation, node.name))
    }
  }  
  const identity = getIdentity()
  glb.gltf.scenes[glb.gltf.scene ?? 0].nodes.forEach(root => dfs(root, identity))
  return bindPose
}

export function getMeshes(gl: WebGL2RenderingContext, glb: GLB, bindPose: Map<string, number[]>): Mesh[] {
  function enableAttribute(location: number, data: any, size: number, type: number, normalized: boolean) {
    const vbo = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW)
    gl.vertexAttribPointer(location, size, type, normalized, 0, 0)
    gl.enableVertexAttribArray(location)
  }

  return glb.gltf.nodes.filter(node => node.mesh !== undefined).map(node => {
    const mesh = glb.gltf.meshes[node.mesh]
    const primitives = mesh.primitives.map(primitive => {
      const vao = gl.createVertexArray()
      gl.bindVertexArray(vao)

      const attributes = primitive.attributes
      if (attributes.POSITION !== undefined) {
        const accessor = glb.access(attributes.POSITION)
        enableAttribute(0, accessor.data, accessor.size, accessor.type, accessor.normalized)
      }
      if (attributes.TEXCOORD_0 !== undefined) {
        const accessor = glb.access(attributes.TEXCOORD_0)
        enableAttribute(1, accessor.data, accessor.size, accessor.type, accessor.normalized)
      }
      if (attributes.NORMAL !== undefined) {
        const accessor = glb.access(attributes.NORMAL)
        enableAttribute(2, accessor.data, accessor.size, accessor.type, accessor.normalized)
      }
      if (attributes.JOINTS_0 !== undefined) {
        const accessor = glb.access(attributes.JOINTS_0)
        enableAttribute(3, accessor.data, accessor.size, accessor.type, accessor.normalized)
      }
      if (attributes.WEIGHTS_0 !== undefined) {
        const accessor = glb.access(attributes.WEIGHTS_0)
        enableAttribute(4, accessor.data, accessor.size, accessor.type, accessor.normalized)
      }
      
      const indices = glb.access(primitive.indices)
      const ebo = gl.createBuffer()
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices.data, gl.STATIC_DRAW)

      const textureId = primitive.material !== undefined && glb.gltf.materials[primitive.material].pbrMetallicRoughness !== undefined
        ? glb.gltf.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index
        : 0

      return new Primitive(vao, indices.count, indices.type, textureId)
    })
    return new Mesh(bindPose.get(node.name), primitives, node.skin)
  })
}

export class Engine {
  constructor(canvasId: string) {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement
    const gl = canvas.getContext("webgl2") as WebGL2RenderingContext
    gl.clearColor(1, 1, 1, 1)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    gl.enable(gl.DEPTH_TEST)
    gl.enable(gl.CULL_FACE)
    this.basicShader = new Shader(gl, new TextureShaderSourceProvider)
    this.characterShader = new Shader(gl, new SkinningMeshShaderSourceProvider)
    this.gl = gl
  }

  setCamera({fov = 45, near = 0.01, far = 1000, position = [0, 0, 30], target = [0, 0, 0], up = [0, 1, 0]}) {
    const canvas = this.gl.canvas as HTMLCanvasElement
    canvas.height = canvas.clientHeight
    canvas.width = canvas.clientWidth
    this.camera = times(perspective(fov, near, far, canvas.width / canvas.height), lookAt(position, target, up))
  }

  async loadBlobTexture(glb: GLB, image: any) {
    const tex = this.gl.createTexture()
    this.gl.bindTexture(this.gl.TEXTURE_2D, tex)
    const bufferView = glb.gltf.bufferViews[image.bufferView]
    const imageBytes = new Uint8Array(glb.bin, bufferView.byteOffset ?? 0, bufferView.byteLength)
    const blob = new Blob([ imageBytes ], { type: image.mimeType })
    const urlCreator = window.URL || window.webkitURL
    const img = new Image()
    img.src = urlCreator.createObjectURL(blob)
    await img.decode()
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, img)
    this.gl.generateMipmap(this.gl.TEXTURE_2D)
    this.gl.bindTexture(this.gl.TEXTURE_2D, null)
    return tex
  }

  async loadKTXTexture(uri: string) {
    const gl = this.gl
    const data = await fetch(`${uri}`).then(response => response.arrayBuffer()).then(arrayBuffer => new Uint32Array(arrayBuffer))
    if (data[0] != 0x58544BAB || data[1] != 0xBB313120 || data[2] != 0x0A1A0A0D) {
      throw new Error("Invalid KTX")
    }
    if (data[3] != 0x04030201) {
      throw new Error("Invalid endianness")
    }
    if (gl.getExtension("WEBGL_compressed_texture_s3tc") == null) {
      throw new Error("Unsupported compressed texture s3tc")
    }
    const texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, 10497)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, 10497)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, 9729)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, 9987)
    const glInternalFormat = data[7]
    const numberOfMipmapLevels = data[14]
    const bytesOfKeyValueData = data[15]
    let pixelWidth = data[9]
    let pixelHeight = data[10]
    let start = 16 + bytesOfKeyValueData / 4
    for (let mipmapLevel = 0; mipmapLevel < numberOfMipmapLevels; ++mipmapLevel) {
      const imageSize = data[start++]
      gl.compressedTexImage2D(gl.TEXTURE_2D, mipmapLevel, glInternalFormat, pixelWidth, pixelHeight, 0, data.slice(start, start + (imageSize >> 2)))
      pixelHeight = pixelHeight >> 1
      pixelWidth = pixelWidth >> 1
      start += imageSize >> 2
    }
    this.gl.bindTexture(this.gl.TEXTURE_2D, null)
    return texture
  }

  async loadStaticModel(uri: string) {
    const glb = await GLB.loadFrom(uri)
    const bindPose = getBindPose(glb)
    const meshes = getMeshes(this.gl, glb, bindPose)
    const textures = new Array<WebGLTexture>()
    for (const image of glb.gltf.images) {
      textures.push(await this.loadBlobTexture(glb, image))
    }
    const model = new Model(meshes, bindPose, textures, getIdentity())
    this.staticModels.push(model)
    return model
  }

  async loadAnimateModel(modelUri: string, skinUri: string) {
    const glb = await GLB.loadFrom(modelUri)
    console.log(glb)
    const bindPose = getBindPose(glb)
    const meshes = getMeshes(this.gl, glb, bindPose)
    const textures = [ await this.loadKTXTexture(skinUri) ]
    console.log(textures)
    const model = new AnimateModel(meshes, bindPose, textures, getIdentity(), glb.gltf.skins, glb.gltf.nodes)
    this.animateModels.push(model)
    return model
  }

  renderStaticModels() {
    const shader = this.basicShader
    const gl = this.gl
    shader.use()
    shader.setMatrix("camera", this.camera)
    this.staticModels.forEach(model => {
      shader.setMatrix("model", model.matrix)
      model.meshes.forEach(mesh => {
        shader.setMatrix("mesh", mesh.matrix)
        mesh.primitives.forEach(primitive => {
          gl.bindTexture(gl.TEXTURE_2D, model.textures[primitive.textureId])
          gl.bindVertexArray(primitive.vao)
          gl.drawElements(gl.TRIANGLES, primitive.count, primitive.type, 0)
        })
      })
    })
  }

  renderAnimateModels(timestamp: number) {
    const shader = this.characterShader
    const gl = this.gl
    shader.use()
    shader.setMatrix("camera", this.camera)
    this.animateModels.forEach(model => {
      shader.setMatrix("model", model.matrix)
      gl.bindTexture(gl.TEXTURE_2D, model.textures[0])
      const animation = model.animations.get(model.currentAnimation)
      const animationPose = animation.getFramePose(timestamp / 1200)
      const skinMats = model.skins.map(skin => {
        return skin.joints.map(joint => {
          const jointName = model.nodes[joint].name
          if (animationPose.has(jointName)) {
            return times(animationPose.get(jointName), model.inverseBindMatrices.get(jointName))
          }
          return getIdentity()
        }).flat()
      })
      model.meshes.forEach(mesh => {
        shader.setMatrix("mesh", mesh.matrix)
        shader.setMatrix("joints", skinMats[mesh.skinIndex])
        mesh.primitives.forEach(primitive => {
          gl.bindVertexArray(primitive.vao)
          gl.drawElements(gl.TRIANGLES, primitive.count, primitive.type, 0)
        })
      })
    })
  }

  renderFrame(timestamp: number) {
    const canvas = this.gl.canvas as HTMLCanvasElement
    canvas.height = canvas.clientHeight
    canvas.width = canvas.clientWidth
    this.gl.viewport(0, 0, canvas.width, canvas.height)
    this.renderStaticModels()
    this.renderAnimateModels(timestamp)
    requestAnimationFrame(timestamp => this.renderFrame(timestamp))
  }

  startLoop() {
    requestAnimationFrame(timestamp => this.renderFrame(timestamp))
  }

  enableClick() {
    const canvas = this.gl.canvas
    const x = canvas.width / 2
    canvas.addEventListener('click', (event) => {
        this.animateModels[0].matrix[14] += 0.005
        const d = Math.PI
        const sin = Math.sin(d)
        const cos = Math.cos(d)
        this.animateModels[0].matrix[0] *= cos
        this.animateModels[0].matrix[2] *= -sin
        this.animateModels[0].matrix[8] *= sin
        this.animateModels[0].matrix[10] *= cos
    })
  }

  gl: WebGL2RenderingContext
  basicShader: Shader
  characterShader: Shader
  staticModels = new Array<Model>()
  animateModels = new Array<AnimateModel>()
  camera: number[]
}