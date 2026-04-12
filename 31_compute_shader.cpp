#include <vulkan/vulkan_core.h>

#include "vulkan/vulkan.hpp"
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// std
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>

constexpr uint32_t WIDTH{800};
constexpr uint32_t HEIGHT{600};
constexpr int FRAMES_IN_FLIGHT{2};
constexpr uint32_t PARTICLE_COUNT{8192};

const std::vector<const char*> validationLayers{"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Particle {
  glm::vec2 position;
  glm::vec2 velocity;
  glm::vec4 color;

  static vk::VertexInputBindingDescription getBindingDescription() {
    return {0, sizeof(Particle), vk::VertexInputRate::eVertex};
  }

  static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
    return {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color)),
    };
  }
};

struct UniformBufferObject {
  float deltaTime = 1.0f;
};

class ComputeShaderApplication {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  GLFWwindow* window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  uint32_t queueIndex = ~0;
  vk::raii::Queue queue = nullptr;
  vk::raii::SwapchainKHR swapchain = nullptr;
  std::vector<vk::Image> swapchainImages;
  vk::SurfaceFormatKHR swapchainSurfaceFormat;
  vk::Extent2D swapchainExtent;
  std::vector<vk::raii::ImageView> swapchainImageViews;

  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;

  vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
  vk::raii::PipelineLayout computePipelineLayout = nullptr;
  vk::raii::Pipeline computePipeline = nullptr;

  std::vector<vk::raii::Buffer> shaderStorageBuffers;
  std::vector<vk::raii::DeviceMemory> shaderStorageBuffersMemory;

  std::vector<vk::raii::Buffer> uniformBuffers;
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
  std::vector<void*> uniformBuffersMapped;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;
  std::vector<vk::raii::CommandBuffer> computeCommandBuffers;

  vk::raii::Semaphore semaphore = nullptr;
  uint64_t timelineValue = 0;
  std::vector<vk::raii::Fence> inFlightFences;
  uint32_t frameIndex = 0;

  bool framebufferResized = false;

  std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};

  double lastFrameTime = 0.0;
  double lastTime = 0.0;

  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    lastTime = glfwGetTime();
  }

  static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<ComputeShaderApplication*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createComputeDescriptorSetLayout();
    createGraphicsPipeline();
    createComputePipeline();
    createCommandPool();
    createShaderStorageBuffers();
    createUniformBuffers();
    createDescriptorPool();
    createComputeDescriptorSets();
    createCommandBuffers();
    createComputeCommandBuffers();
    createSyncObjects();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
      double currentTime = glfwGetTime();
      lastFrameTime = (currentTime - lastTime) * 1000.0;
      lastTime = currentTime;
      std::cout << "Frame Time: " << lastFrameTime << " ms\n";
    }

    device.waitIdle();
  }

  void cleanupSwapchain() {
    swapchainImageViews.clear();
    swapchain = nullptr;
  }

  void cleanup() { cleanupSwapchain(); }

  void recreateSwapchain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }
    device.waitIdle();

    cleanupSwapchain();

    createSwapchain();
    createImageViews();
  }

  void createInstance() {
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = vk::ApiVersion14,
    };

    const auto requiredExtensions = getRequiredInstanceExtensions();
    const auto extensionProperties = context.enumerateInstanceExtensionProperties();
    const auto unsupportedPropertyIt =
        std::ranges::find_if(requiredExtensions, [&extensionProperties](const auto& requiredExtension) {
          return std::ranges::none_of(extensionProperties, [requiredExtension](const auto& extensionProperty) {
            return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
          });
        });

    if (unsupportedPropertyIt != requiredExtensions.end()) {
      throw std::runtime_error("Required extension not supported" + std::string(*unsupportedPropertyIt));
    }

    std::cout << "available extensions:\n";
    for (const auto& extension : extensionProperties) {
      std::cout << '\t' << extension.extensionName << '\n';
    }

    std::vector<const char*> requiredLayers;
    if (enableValidationLayers) {
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    const auto layerProperties = context.enumerateInstanceLayerProperties();
    const auto unsupportedLayerIt = std::ranges::find_if(requiredLayers, [&layerProperties](const auto& requiredLayer) {
      return std::ranges::none_of(layerProperties, [requiredLayer](const auto& layerProperty) {
        return strcmp(layerProperty.layerName, requiredLayer) == 0;
      });
    });

    if (unsupportedLayerIt != requiredLayers.end()) {
      throw std::runtime_error("Required layer not supported" + std::string(*unsupportedLayerIt));
    }

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
        .ppEnabledLayerNames = requiredLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
        .ppEnabledExtensionNames = requiredExtensions.data(),
    };

    instance = vk::raii::Instance(context, createInfo);
  }
  //
  void setupDebugMessenger() {
    if (!enableValidationLayers) return;

    vk::DebugUtilsMessageSeverityFlagsEXT severtityFlags{vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eError};
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags{vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation};
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCI{
        .messageSeverity = severtityFlags,
        .messageType = messageTypeFlags,
        .pfnUserCallback = &debugCallback,
    };

    debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCI);
  }

  void createSurface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != VK_SUCCESS) {
      const char* buf = (const char*)calloc(512, sizeof(char));
      int result = glfwGetError(&buf);
      std::cerr << std::string(buf) << std::endl;
      free((void*)buf);
      throw std::runtime_error("failed to create window surface");
    }

    surface = vk::raii::SurfaceKHR(instance, _surface);
  }

  bool isDeviceSuitable(const vk::raii::PhysicalDevice& physicalDevice) {
    const auto deviceProperties = physicalDevice.getProperties();
    const auto deviceFeatures = physicalDevice.getFeatures();

    // Check if the physical device supports the Vulkan 1.3 API version
    if (deviceProperties.apiVersion < vk::ApiVersion13) return false;

    // Check if any of the queue families support the graphics operations
    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    if (std::ranges::none_of(queueFamilies,
                             [](const auto& qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); }))
      return false;

    // Check if all of the required device extensions are available
    std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};
    const auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    if (std::ranges::any_of(requiredDeviceExtension, [&availableDeviceExtensions](const auto& requiredDeviceExtension) {
          return std::ranges::none_of(availableDeviceExtensions,
                                      [requiredDeviceExtension](const auto& availableExtension) {
                                        return strcmp(availableExtension.extensionName, requiredDeviceExtension) == 0;
                                      });
        })) {
      return false;
    }

    // Check if the physical device supports the required features (dynamic rendering and extended dynamic state)
    const auto features =
        physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                                             vk::PhysicalDeviceVulkan13Features,
                                             vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    if (!(features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
          features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
          features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
          features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
          features.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy)) {
      return false;
    }

    return true;
  }

  void pickPhysicalDevice() {
    const auto physicalDevices = instance.enumeratePhysicalDevices();

    if (physicalDevices.empty()) {
      throw std::runtime_error("failed to find GPUs with Vulkan support");
    }

    const auto deviceIt = std::ranges::find_if(physicalDevices, [&](const auto& pd) { return isDeviceSuitable(pd); });
    if (deviceIt == physicalDevices.end()) {
      throw std::runtime_error("failed to find suitable device");
    }

    physicalDevice = *deviceIt;
  }

  void createLogicalDevice() {
    const std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    for (auto qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
      if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
          (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
          physicalDevice.getSurfaceSupportKHR(qfpIndex, surface)) {
        queueIndex = qfpIndex;
        break;
      }
    }

    if (queueIndex == ~0) {
      throw std::runtime_error("Could not find a queue for graphics and present");
    }

    const float queuePriority = 0.5f;
    const vk::DeviceQueueCreateInfo deviceQueueCI{
        .queueFamilyIndex = queueIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT, vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>
        featureChain = {
            {.features = {.samplerAnisotropy = true}},
            {.synchronization2 = true, .dynamicRendering = true},
            {.extendedDynamicState = true},
            {.shaderDrawParameters = true},
            {.timelineSemaphore = true},
        };

    std::vector<const char*> requiredDeviceExtensions = {vk::KHRSwapchainExtensionName};

    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
    std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);

    const vk::DeviceCreateInfo deviceCI{
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCI,
        .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions.size()),
        .ppEnabledExtensionNames = requiredDeviceExtensions.data(),
    };

    device = vk::raii::Device(physicalDevice, deviceCI);
    queue = vk::raii::Queue(device, queueIndex, 0);
  }

  void createSwapchain() {
    vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapchainExtent = chooseSwapExtent(surfaceCapabilities);
    const uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
    swapchainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

    const std::vector<vk::PresentModeKHR> availableModes = physicalDevice.getSurfacePresentModesKHR(*surface);
    const auto swapchainPresentMode = chooseSwapPresentMode(availableModes);

    const vk::SwapchainCreateInfoKHR swapchainCI{
        .surface = *surface,
        .minImageCount = minImageCount,
        .imageFormat = swapchainSurfaceFormat.format,
        .imageColorSpace = swapchainSurfaceFormat.colorSpace,
        .imageExtent = swapchainExtent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = surfaceCapabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = swapchainPresentMode,
        .clipped = true,
    };

    swapchain = vk::raii::SwapchainKHR(device, swapchainCI);
    swapchainImages = swapchain.getImages();
  }

  void createImageViews() {
    assert(swapchainImageViews.empty());

    for (auto image : swapchainImages) {
      const vk::ImageViewCreateInfo imageViewCI{
          .image = image,
          .viewType = vk::ImageViewType::e2D,
          .format = swapchainSurfaceFormat.format,
          .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
      };

      swapchainImageViews.emplace_back(vk::raii::ImageView(device, imageViewCI));
    }
  }

  void createComputeDescriptorSetLayout() {
    std::array bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute,
                                       nullptr),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute,
                                       nullptr),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute,
                                       nullptr),
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
  }

  void createGraphicsPipeline() {
    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    auto bindingDescription = Particle::getBindingDescription();
    auto attributeDescriptions = Particle::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data()};
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::ePointList,
                                                           .primitiveRestartEnable = vk::False};
    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable = vk::False,
                                                        .rasterizerDiscardEnable = vk::False,
                                                        .polygonMode = vk::PolygonMode::eFill,
                                                        .cullMode = vk::CullModeFlagBits::eBack,
                                                        .frontFace = vk::FrontFace::eCounterClockwise,
                                                        .depthBiasEnable = vk::False,
                                                        .lineWidth = 1.0f};
    vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1,
                                                         .sampleShadingEnable = vk::False};

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False,
                                                        .logicOp = vk::LogicOp::eCopy,
                                                        .attachmentCount = 1,
                                                        .pAttachments = &colorBlendAttachment};

    std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
                                                    .pDynamicStates = dynamicStates.data()};

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        {.stageCount = 2,
         .pStages = shaderStages,
         .pVertexInputState = &vertexInputInfo,
         .pInputAssemblyState = &inputAssembly,
         .pViewportState = &viewportState,
         .pRasterizationState = &rasterizer,
         .pMultisampleState = &multisampling,
         .pColorBlendState = &colorBlending,
         .pDynamicState = &dynamicState,
         .layout = pipelineLayout,
         .renderPass = nullptr},
        {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchainSurfaceFormat.format}};

    graphicsPipeline =
        vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }
  // void createGraphicsPipeline() {
  //   const vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));
  //   const vk::PipelineShaderStageCreateInfo vertexShaderCI{
  //       .stage = vk::ShaderStageFlagBits::eVertex,
  //       .module = shaderModule,
  //       .pName = "vertMain",
  //   };
  //   const vk::PipelineShaderStageCreateInfo fragmentShaderCI{
  //       .stage = vk::ShaderStageFlagBits::eFragment,
  //       .module = shaderModule,
  //       .pName = "fragMain",
  //   };
  //
  //   vk::PipelineShaderStageCreateInfo shaderStageCIs[] = {vertexShaderCI, fragmentShaderCI};
  //
  //   const auto vertexBindingDescription = Particle::getBindingDescription();
  //   const auto vertexAttributeDescriptions = Particle::getAttributeDescriptions();
  //   vk::PipelineVertexInputStateCreateInfo vertexInputCI{
  //       .vertexBindingDescriptionCount = 1,
  //       .pVertexBindingDescriptions = &vertexBindingDescription,
  //       .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size()),
  //       .pVertexAttributeDescriptions = vertexAttributeDescriptions.data(),
  //   };
  //   vk::PipelineInputAssemblyStateCreateInfo inputAssemblyCI{.topology = vk::PrimitiveTopology::eTriangleList};
  //
  //   vk::Viewport viewport{
  //       .x = 0.0f,
  //       .y = 0.0f,
  //       .width = static_cast<float>(swapchainExtent.width),
  //       .height = static_cast<float>(swapchainExtent.height),
  //       .minDepth = 0.0f,
  //       .maxDepth = 1.0f,
  //   };
  //
  //   vk::Rect2D scissor{vk::Offset2D{0, 0}, swapchainExtent};
  //
  //   std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
  //   vk::PipelineDynamicStateCreateInfo dynamicStateCI{
  //       .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
  //       .pDynamicStates = dynamicStates.data(),
  //   };
  //   vk::PipelineViewportStateCreateInfo viewportCI{
  //       .viewportCount = 1,
  //       .scissorCount = 1,
  //   };
  //
  //   vk::PipelineRasterizationStateCreateInfo rasterizationCI{
  //       .depthClampEnable = vk::False,
  //       .rasterizerDiscardEnable = vk::False,
  //       .polygonMode = vk::PolygonMode::eFill,
  //       .cullMode = vk::CullModeFlagBits::eBack,
  //       .frontFace = vk::FrontFace::eCounterClockwise,
  //       .depthBiasEnable = vk::False,
  //       .depthBiasSlopeFactor = 1.0f,
  //       .lineWidth = 1.0f,
  //   };
  //
  //   vk::PipelineMultisampleStateCreateInfo multisampleCI{
  //       .rasterizationSamples = vk::SampleCountFlagBits::e1,
  //       .sampleShadingEnable = vk::False,
  //   };
  //
  //   vk::PipelineColorBlendAttachmentState colorBlendAttachment{
  //       .blendEnable = vk::False,
  //       .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
  //                         vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  //   };
  //
  //   vk::PipelineDepthStencilStateCreateInfo depthStencilStateCI{
  //       .depthTestEnable = vk::True,
  //       .depthWriteEnable = vk::True,
  //       .depthCompareOp = vk::CompareOp::eLess,
  //       .depthBoundsTestEnable = vk::False,
  //       .stencilTestEnable = vk::False,
  //   };
  //
  //   vk::PipelineColorBlendStateCreateInfo colorBlendCI{
  //       .logicOpEnable = vk::False,
  //       .logicOp = vk::LogicOp::eCopy,
  //       .attachmentCount = 1,
  //       .pAttachments = &colorBlendAttachment,
  //   };
  //
  //   vk::PipelineLayoutCreateInfo pipelineLayoutCI{};
  //
  //   pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
  //
  //   vk::Format depthFormat = findDepthFormat();
  //
  //   vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
  //       {
  //           .stageCount = 2,
  //           .pStages = shaderStageCIs,
  //           .pVertexInputState = &vertexInputCI,
  //           .pInputAssemblyState = &inputAssemblyCI,
  //           .pViewportState = &viewportCI,
  //           .pRasterizationState = &rasterizationCI,
  //           .pMultisampleState = &multisampleCI,
  //           .pDepthStencilState = &depthStencilStateCI,
  //           .pColorBlendState = &colorBlendCI,
  //           .pDynamicState = &dynamicStateCI,
  //           .layout = pipelineLayout,
  //           .renderPass = nullptr,
  //       },
  //       {
  //           .colorAttachmentCount = 1,
  //           .pColorAttachmentFormats = &swapchainSurfaceFormat.format,
  //           .depthAttachmentFormat = depthFormat,
  //       },
  //   };
  //
  //   graphicsPipeline =
  //       vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  // }

  void createComputePipeline() {
    const vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));
    const vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = shaderModule,
        .pName = "compMain",
    };
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*computeDescriptorSetLayout,
    };

    computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::ComputePipelineCreateInfo pipelineInfo{
        .stage = computeShaderStageInfo,
        .layout = *computePipelineLayout,
    };

    computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
  }

  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = queueIndex;
    commandPool = vk::raii::CommandPool(device, poolInfo);
  }

  void createShaderStorageBuffers() {
    std::default_random_engine rndEngine((unsigned)time(nullptr));
    std::uniform_real_distribution rndDist(0.0f, 1.0f);

    std::vector<Particle> particles(PARTICLE_COUNT);
    for (auto& particle : particles) {
      float r = 0.25f * sqrtf(rndDist(rndEngine));
      float theta = rndDist(rndEngine) * 2.0f * glm::pi<float>();
      float x = r * cosf(theta) * HEIGHT / WIDTH;
      float y = r * sinf(theta);
      particle.position = glm::vec2(x, y);
      particle.velocity = glm::normalize(particle.position) * 0.00025f;
      particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
    }

    vk::DeviceSize bufferSize = particles.size() * sizeof(Particle);

    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingBufferMemory = nullptr;
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer,
                 stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, particles.data(), static_cast<size_t>(bufferSize));
    stagingBufferMemory.unmapMemory();

    for (auto i = 0; i < FRAMES_IN_FLIGHT; i++) {
      vk::raii::Buffer tempBuffer = nullptr;
      vk::raii::DeviceMemory tempBufferMemory = nullptr;

      createBuffer(bufferSize,
                   vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
                       vk::BufferUsageFlagBits::eTransferDst,
                   vk::MemoryPropertyFlagBits::eDeviceLocal, tempBuffer, tempBufferMemory);

      copyBuffer(stagingBuffer, tempBuffer, bufferSize);
      shaderStorageBuffers.emplace_back(std::move(tempBuffer));
      shaderStorageBuffersMemory.emplace_back(std::move(tempBufferMemory));
    }
  }

  void createUniformBuffers() {
    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();

    for (auto i = 0; i < FRAMES_IN_FLIGHT; i++) {
      vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
      vk::raii::Buffer buffer = nullptr;
      vk::raii::DeviceMemory bufferMemory = nullptr;
      createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer,
                   bufferMemory);

      uniformBuffers.emplace_back(std::move(buffer));
      uniformBuffersMemory.emplace_back(std::move(bufferMemory));
      uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
    }
  }

  void createDescriptorPool() {
    std::array poolSizes{
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = FRAMES_IN_FLIGHT,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 2 * FRAMES_IN_FLIGHT,
        },
    };

    vk::DescriptorPoolCreateInfo poolCI{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = FRAMES_IN_FLIGHT,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    descriptorPool = vk::raii::DescriptorPool(device, poolCI);
  }

  void createComputeDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo descriptorSetAI{
        .descriptorPool = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
    };

    computeDescriptorSets.clear();
    computeDescriptorSets = device.allocateDescriptorSets(descriptorSetAI);

    for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo descriptorBufferInfo{
          .buffer = uniformBuffers[i],
          .offset = 0,
          .range = sizeof(UniformBufferObject),
      };

      vk::DescriptorBufferInfo storageBufferInfoLastFrame(
          shaderStorageBuffers[(i + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT], 0, sizeof(Particle) * PARTICLE_COUNT);
      vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(shaderStorageBuffers[i], 0,
                                                             sizeof(Particle) * PARTICLE_COUNT);

      std::array writeDescriptorSets{
          vk::WriteDescriptorSet{
              .dstSet = computeDescriptorSets[i],
              .dstBinding = 0,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer,
              .pBufferInfo = &descriptorBufferInfo,
          },
          vk::WriteDescriptorSet{
              .dstSet = computeDescriptorSets[i],
              .dstBinding = 1,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo = &storageBufferInfoLastFrame,
          },
          vk::WriteDescriptorSet{
              .dstSet = computeDescriptorSets[i],
              .dstBinding = 2,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo = &storageBufferInfoCurrentFrame,
          },
      };

      device.updateDescriptorSets(writeDescriptorSets, {});
    }
  }

  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                    vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
    vk::BufferCreateInfo bufferCI{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    buffer = vk::raii::Buffer(device, bufferCI);

    vk::MemoryRequirements memoryReqs = buffer.getMemoryRequirements();

    vk::MemoryAllocateInfo memoryAI{
        .allocationSize = memoryReqs.size,
        .memoryTypeIndex = findMemoryType(memoryReqs.memoryTypeBits, properties),
    };

    bufferMemory = vk::raii::DeviceMemory(device, memoryAI);
    buffer.bindMemory(*bufferMemory, 0);
  }

  [[nodiscard]] vk::raii::CommandBuffer beginSingleTimeCommands() const {
    vk::CommandBufferAllocateInfo commandBufferAllocInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    vk::raii::CommandBuffer commandBuffer = std::move(device.allocateCommandBuffers(commandBufferAllocInfo).front());

    vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    commandBuffer.begin(beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) const {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffer,
    };

    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
  }

  void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
    auto commandBuffer = beginSingleTimeCommands();
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
    endSingleTimeCommands(commandBuffer);
  }

  [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    const auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type");
  }

  void createCommandBuffers() {
    vk::CommandBufferAllocateInfo commandBufferAI{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = FRAMES_IN_FLIGHT,
    };

    commandBuffers = vk::raii::CommandBuffers(device, commandBufferAI);
  }

  void createComputeCommandBuffers() {
    vk::CommandBufferAllocateInfo computeCommandBufferInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = FRAMES_IN_FLIGHT,
    };

    computeCommandBuffers = vk::raii::CommandBuffers(device, computeCommandBufferInfo);
  }

  void recordCommandBuffer(uint32_t imageIndex) {
    auto& commandBuffer = commandBuffers[frameIndex];
    commandBuffer.reset();
    commandBuffer.begin({});
    // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
    transition_image_layout(imageIndex, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                            {},  // srcAccessMask (no need to wait for previous operations)
                            vk::AccessFlagBits2::eColorAttachmentWrite,          // dstAccessMask
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,  // srcStage
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput   // dstStage
    );
    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::RenderingAttachmentInfo attachmentInfo = {.imageView = swapchainImageViews[imageIndex],
                                                  .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                                  .loadOp = vk::AttachmentLoadOp::eClear,
                                                  .storeOp = vk::AttachmentStoreOp::eStore,
                                                  .clearValue = clearColor};
    vk::RenderingInfo renderingInfo = {.renderArea = {.offset = {0, 0}, .extent = swapchainExtent},
                                       .layerCount = 1,
                                       .colorAttachmentCount = 1,
                                       .pColorAttachments = &attachmentInfo};
    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width),
                                              static_cast<float>(swapchainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));
    commandBuffer.bindVertexBuffers(0, {shaderStorageBuffers[frameIndex]}, {0});
    commandBuffer.draw(PARTICLE_COUNT, 1, 0, 0);
    commandBuffer.endRendering();
    // After rendering, transition the swapchain image to PRESENT_SRC
    transition_image_layout(imageIndex, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                            vk::AccessFlagBits2::eColorAttachmentWrite,          // srcAccessMask
                            {},                                                  // dstAccessMask
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,  // srcStage
                            vk::PipelineStageFlagBits2::eBottomOfPipe            // dstStage
    );
    commandBuffer.end();
  }
  // void recordCommandBuffer(uint32_t imageIndex) {
  //   commandBuffers[frameIndex].reset();
  //   commandBuffers[frameIndex].begin({});
  //
  //
  //   transition_image_layout(swapchainImages[imageIndex], vk::ImageLayout::eUndefined,
  //                           vk::ImageLayout::eColorAttachmentOptimal, {}, vk::AccessFlagBits2::eColorAttachmentWrite,
  //                           vk::PipelineStageFlagBits2::eColorAttachmentOutput,
  //                           vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
  //
  //   transition_image_layout(*colorImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
  //                           vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eColorAttachmentWrite,
  //                           vk::PipelineStageFlagBits2::eColorAttachmentOutput,
  //                           vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
  //
  //   vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
  //
  //   vk::RenderingAttachmentInfo renderingAttachmentInfo{
  //       .imageView = colorImageView,
  //       .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
  //       .resolveMode = vk::ResolveModeFlagBits::eNone,
  //       .resolveImageView = swapchainImageViews[imageIndex],
  //       .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
  //       .loadOp = vk::AttachmentLoadOp::eClear,
  //       .storeOp = vk::AttachmentStoreOp::eStore,
  //       .clearValue = clearColor,
  //   };
  //
  //   vk::RenderingInfo renderingInfo{
  //       .renderArea = {.offset = {0, 0}, .extent = swapchainExtent},
  //       .layerCount = 1,
  //       .colorAttachmentCount = 1,
  //       .pColorAttachments = &renderingAttachmentInfo,
  //   };
  //
  //   commandBuffers[frameIndex].beginRendering(renderingInfo);
  //
  //   commandBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
  //   commandBuffers[frameIndex].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width),
  //                                                          static_cast<float>(swapchainExtent.height), 0.0f, 1.0f));
  //   commandBuffers[frameIndex].setScissor(0, vk::Rect2D{vk::Offset2D{0, 0}, swapchainExtent});
  //
  //   commandBuffers[frameIndex].bindVertexBuffers(0, {*shaderStorageBuffers[frameIndex]}, {0});
  //   commandBuffers[frameIndex].draw(PARTICLE_COUNT, 1, 0, 0);
  //
  //   commandBuffers[frameIndex].endRendering();
  //
  //   transition_image_layout(swapchainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
  //                           vk::ImageLayout::ePresentSrcKHR, vk::AccessFlagBits2::eColorAttachmentWrite, {},
  //                           vk::PipelineStageFlagBits2::eColorAttachmentOutput,
  //                           vk::PipelineStageFlagBits2::eBottomOfPipe, vk::ImageAspectFlagBits::eColor);
  //
  //   commandBuffers[frameIndex].end();
  // }
  //
  void transition_image_layout(uint32_t imageIndex, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                               vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
                               vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask) {
    vk::ImageMemoryBarrier2 barrier = {.srcStageMask = src_stage_mask,
                                       .srcAccessMask = src_access_mask,
                                       .dstStageMask = dst_stage_mask,
                                       .dstAccessMask = dst_access_mask,
                                       .oldLayout = old_layout,
                                       .newLayout = new_layout,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .image = swapchainImages[imageIndex],
                                       .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                            .baseMipLevel = 0,
                                                            .levelCount = 1,
                                                            .baseArrayLayer = 0,
                                                            .layerCount = 1}};
    vk::DependencyInfo dependency_info = {
        .dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
    commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
  }
  // void transition_image_layout(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
  //                              vk::AccessFlags2 srcAccessMask, vk::AccessFlags2 dstAccessMask,
  //                              vk::PipelineStageFlags2 srcStageMask, vk::PipelineStageFlags2 dstStageMask,
  //                              vk::ImageAspectFlags aspectMask) {
  //   vk::ImageMemoryBarrier2 barrier{
  //       .srcStageMask = srcStageMask,
  //       .srcAccessMask = srcAccessMask,
  //       .dstStageMask = dstStageMask,
  //       .dstAccessMask = dstAccessMask,
  //       .oldLayout = oldLayout,
  //       .newLayout = newLayout,
  //       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
  //       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
  //       .image = image,
  //       .subresourceRange =
  //           {
  //               .aspectMask = aspectMask,
  //               .baseMipLevel = 0,
  //               .levelCount = 1,
  //               .baseArrayLayer = 0,
  //               .layerCount = 1,
  //           },
  //   };
  //
  //   vk::DependencyInfo dependencyInfo{
  //       .dependencyFlags = {},
  //       .imageMemoryBarrierCount = 1,
  //       .pImageMemoryBarriers = &barrier,
  //   };
  //
  //   commandBuffers[frameIndex].pipelineBarrier2(dependencyInfo);
  // }

  void recordComputeCommandBuffer() {
    computeCommandBuffers[frameIndex].reset();
    computeCommandBuffers[frameIndex].begin({});
    computeCommandBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
    computeCommandBuffers[frameIndex].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0,
                                                         {computeDescriptorSets[frameIndex]}, {});

    computeCommandBuffers[frameIndex].dispatch(PARTICLE_COUNT / 256, 1, 1);

    computeCommandBuffers[frameIndex].end();
  }

  void createSyncObjects() {
    assert(inFlightFences.empty());

    vk::SemaphoreTypeCreateInfo semaphoreType{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = 0,
    };
    semaphore = vk::raii::Semaphore(device, {.pNext = &semaphoreType});
    timelineValue = 0;

    for (auto i = 0; i < FRAMES_IN_FLIGHT; i++) {
      inFlightFences.emplace_back(device, vk::FenceCreateInfo{});
    }
  }

  void updateUniformBuffer(uint32_t currentImage) {
    UniformBufferObject ubo{.deltaTime = static_cast<float>(lastFrameTime) * 2.0f};

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
  }

  void drawFrame() {
    auto [result, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, nullptr, *inFlightFences[frameIndex]);
    auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for fence");
    }
    device.resetFences(*inFlightFences[frameIndex]);

    uint64_t computeWaitValue = timelineValue;
    uint64_t computeSignalValue = ++timelineValue;
    uint64_t graphicsWaitValue = computeSignalValue;
    uint64_t graphicsSignalValue = ++timelineValue;

    updateUniformBuffer(frameIndex);

    {
      recordComputeCommandBuffer();
      vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{
          .waitSemaphoreValueCount = 1,
          .pWaitSemaphoreValues = &computeWaitValue,
          .signalSemaphoreValueCount = 1,
          .pSignalSemaphoreValues = &computeSignalValue,
      };

      vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eComputeShader};

      const vk::SubmitInfo computeSubmitInfo{
          .pNext = &computeTimelineInfo,
          .waitSemaphoreCount = 1,
          .pWaitSemaphores = &*semaphore,
          .pWaitDstStageMask = waitStages,
          .commandBufferCount = 1,
          .pCommandBuffers = &*computeCommandBuffers[frameIndex],
          .signalSemaphoreCount = 1,
          .pSignalSemaphores = &*semaphore,
      };

      queue.submit(computeSubmitInfo, nullptr);
    }

    {
      recordCommandBuffer(imageIndex);
      vk::PipelineStageFlags waitDestinationStageMask{vk::PipelineStageFlagBits::eVertexInput};

      vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
          .waitSemaphoreValueCount = 1,
          .pWaitSemaphoreValues = &graphicsWaitValue,
          .signalSemaphoreValueCount = 1,
          .pSignalSemaphoreValues = &graphicsSignalValue,
      };

      vk::SubmitInfo graphicsSubmitInfo{
          .pNext = &graphicsTimelineInfo,
          .waitSemaphoreCount = 1,
          .pWaitSemaphores = &*semaphore,
          .pWaitDstStageMask = &waitDestinationStageMask,
          .commandBufferCount = 1,
          .pCommandBuffers = &*commandBuffers[frameIndex],
          .signalSemaphoreCount = 1,
          .pSignalSemaphores = &*semaphore,
      };

      queue.submit(graphicsSubmitInfo, nullptr);

      vk::SemaphoreWaitInfo waitInfo{
          .semaphoreCount = 1,
          .pSemaphores = &*semaphore,
          .pValues = &graphicsSignalValue,
      };

      result = device.waitSemaphores(waitInfo, UINT64_MAX);
      if (result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to wait for semaphore");
      }

      const vk::PresentInfoKHR presentInfo{
          .waitSemaphoreCount = 0,
          .pWaitSemaphores = nullptr,
          .swapchainCount = 1,
          .pSwapchains = &*swapchain,
          .pImageIndices = &imageIndex,
      };

      result = queue.presentKHR(presentInfo);
      if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapchain();
      }
    }

    frameIndex = (frameIndex + 1) % FRAMES_IN_FLIGHT;
  }

  [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& shaderCode) const {
    vk::ShaderModuleCreateInfo shaderModuleCI{
        .codeSize = shaderCode.size(),
        .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data()),
    };

    vk::raii::ShaderModule shaderModule{device, shaderModuleCI};

    return shaderModule;
  }

  uint32_t chooseSwapMinImageCount(const vk::SurfaceCapabilitiesKHR& surfaceCapabilities) {
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if (minImageCount < surfaceCapabilities.maxImageCount) {
      minImageCount = surfaceCapabilities.maxImageCount;
    }

    return minImageCount;
  }

  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    assert(!availableFormats.empty());
    const auto it = std::ranges::find_if(availableFormats, [](const auto& fmt) {
      return fmt.format == vk::Format::eB8G8R8A8Srgb && fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });

    if (it != availableFormats.end()) return *it;

    return availableFormats[0];
  }

  static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    assert(
        std::ranges::any_of(availablePresentModes, [](const auto& mode) { return mode == vk::PresentModeKHR::eFifo;
        }));

    if (std::ranges::any_of(availablePresentModes,
                            [](const auto& mode) { return mode == vk::PresentModeKHR::eMailbox; })) {
      return vk::PresentModeKHR::eMailbox;
    }

    return vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    return {
        std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
  }

  std::vector<const char*> getRequiredInstanceExtensions() {
    uint32_t glfwExtensionCount{0};
    const auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
  }

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                                        vk::DebugUtilsMessageTypeFlagsEXT type,
                                                        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void* pUserData) {
    std::cerr << "validation layer: type" << to_string(type) << "msg" << pCallbackData->pMessage << std::endl;

    return vk::False;
  }

  static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file: " + filename);
    }

    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);

    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));

    file.close();

    return buffer;
  }
};

int main() {
  if (!glfwInit()) {
    std::cerr << "failed to initialize glfw";
    return EXIT_FAILURE;
  }

  try {
    ComputeShaderApplication app;
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    glfwTerminate();
    return EXIT_FAILURE;
  }

  glfwTerminate();

  return EXIT_SUCCESS;
}
