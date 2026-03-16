#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <ostream>

#include "vulkan/vulkan.hpp"
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

constexpr uint32_t WIDTH{800};
constexpr uint32_t HEIGHT{800};

const std::vector<const char*> validationLayers{"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

class HelloTriangleApplication {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  GLFWwindow* window;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  vk::raii::Queue graphicsQueue = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::SwapchainKHR swapchain = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;
  std::vector<vk::Image> swapchainImages;
  vk::SurfaceFormatKHR swapchainSurfaceFormat;
  vk::Extent2D swapchainExtent;
  std::vector<vk::raii::ImageView> swapchainImageViews;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

  void initWindow() {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    if (!glfwInit()) {
      throw std::runtime_error("glfw failed to init");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createGraphicsPipeline();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
    glfwDestroyWindow(window);
    glfwTerminate();
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

  std::vector<const char*> getRequiredInstanceExtensions() {
    uint32_t glfwExtensionCount{0};
    const auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
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
        physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
                                             vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    if (!(features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
          features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState)) {
      return false;
    }

    return true;
  }

  void createLogicalDevice() {
    const std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    uint32_t queueIndex = ~0;
    for (auto qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
      if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
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

    vk::PhysicalDeviceFeatures deviceFeatures;
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT, vk::PhysicalDeviceVulkan11Features>
        featureChain = {
            {},
            {.dynamicRendering = true},
            {.extendedDynamicState = true},
            {.shaderDrawParameters = true},
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
    graphicsQueue = vk::raii::Queue(device, queueIndex, 0);
  }

  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    assert(!availableFormats.empty());
    const auto it = std::ranges::find_if(availableFormats, [](const auto& fmt) {
      return fmt.format == vk::Format::eB8G8R8A8Srgb && fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });

    if (it != availableFormats.end()) return *it;

    return availableFormats[0];
  }

  vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availableModes) {
    assert(std::ranges::any_of(availableModes, [](const auto& mode) { return mode == vk::PresentModeKHR::eFifo; }));

    if (std::ranges::any_of(availableModes, [](const auto& mode) { return mode == vk::PresentModeKHR::eMailbox; })) {
      return vk::PresentModeKHR::eMailbox;
    }

    return vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilites) {
    if (capabilites.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilites.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    return {
        std::clamp<uint32_t>(width, capabilites.minImageExtent.width, capabilites.maxImageExtent.width),
        std::clamp<uint32_t>(height, capabilites.minImageExtent.height, capabilites.maxImageExtent.height),
    };
  }

  uint32_t chooseSwapMinImageCount(const vk::SurfaceCapabilitiesKHR& surfaceCapabilities) {
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if (minImageCount < surfaceCapabilities.maxImageCount) {
      minImageCount = surfaceCapabilities.maxImageCount;
    }

    return minImageCount;
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

  void createGraphicsPipeline() {
    const vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));
    const vk::PipelineShaderStageCreateInfo vertexShaderCI{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shaderModule,
        .pName = "vertMain",
    };
    const vk::PipelineShaderStageCreateInfo fragmentShaderCI{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shaderModule,
        .pName = "fragMain",
    };

    vk::PipelineShaderStageCreateInfo shaderStageCIs[] = {vertexShaderCI, fragmentShaderCI};

    vk::PipelineVertexInputStateCreateInfo vertexInputCI{};
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyCI{.topology = vk::PrimitiveTopology::eTriangleList};

    vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(swapchainExtent.width),
        .height = static_cast<float>(swapchainExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    vk::Rect2D scissor{vk::Offset2D{0, 0}, swapchainExtent};

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicStateCI{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };
    vk::PipelineViewportStateCreateInfo viewportCI{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizationCI{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = vk::False,
        .depthBiasSlopeFactor = 1.0f,
        .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampleCI{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = vk::False,
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlendCI{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutCI{
        .setLayoutCount = 0,
        .pushConstantRangeCount = 0,
    };

    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);

    vk::PipelineRenderingCreateInfo pipelineRenderingCI{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapchainSurfaceFormat.format,
    };

    vk::GraphicsPipelineCreateInfo graphicsPipelineCI{
        .pNext = &pipelineRenderingCI,
        .stageCount = 2,
        .pStages = shaderStageCIs,
        .pVertexInputState = &vertexInputCI,
        .pInputAssemblyState = &inputAssemblyCI,
        .pViewportState = &viewportCI,
        .pRasterizationState = &rasterizationCI,
        .pMultisampleState = &multisampleCI,
        .pColorBlendState = &colorBlendCI,
        .pDynamicState = &dynamicStateCI,
        .layout = pipelineLayout,
        .renderPass = nullptr,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = -1,
    };

    graphicsPipeline = vk::raii::Pipeline(device, nullptr, graphicsPipelineCI);
  }

  [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& shaderCode) const {
    vk::ShaderModuleCreateInfo shaderModuleCI{
        .codeSize = shaderCode.size(),
        .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data()),
    };

    vk::raii::ShaderModule shaderModule{device, shaderModuleCI};

    return shaderModule;
  }

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                                        vk::DebugUtilsMessageTypeFlagsEXT type,
                                                        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void* pUserData) {
    std::cerr << "validation layer: type" << to_string(type) << "msg" << pCallbackData->pMessage << std::endl;

    return vk::False;
  }

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
  try {
    HelloTriangleApplication app;
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
