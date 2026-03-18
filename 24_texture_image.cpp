#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
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

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

// std
#include <cstdlib>
#include <iostream>
#include <stdexcept>

constexpr uint32_t WIDTH{800};
constexpr uint32_t HEIGHT{800};
constexpr int FRAMES_IN_FLIGHT{2};

const std::vector<const char*> validationLayers{"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
  glm::vec2 position;
  glm::vec3 color;

  static vk::VertexInputBindingDescription getBindingDescription() {
    return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
  }

  static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescription() {
    return {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, position)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
    };
  }
};

struct ShaderData {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 projection;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

class Window {
 public:
  Window(std::nullptr_t) : handle{} {}
  Window(const char* title, int width, int height) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    handle = glfwCreateWindow(width, height, title, nullptr, nullptr);
  }

  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;

  Window(Window&& other) noexcept : handle(other.handle) { other.handle = nullptr; }
  Window& operator=(Window&& other) noexcept {
    if (this != &other) {
      handle = other.handle;
      other.handle = nullptr;
    }

    return *this;
  }

  ~Window() {
    if (handle != nullptr) {
      glfwDestroyWindow(handle);
    }
  }

  void setFramebuffferSizeCallback(GLFWframebuffersizefun callback) {
    glfwSetFramebufferSizeCallback(handle, callback);
  }

  void setUserPointer(void* ptr) { glfwSetWindowUserPointer(handle, ptr); }

  GLFWwindow* handle;
};

class HelloTriangleApplication {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  Window window = nullptr;
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

  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;

  vk::raii::Buffer vertexBuffer = nullptr;
  vk::raii::DeviceMemory vertexBufferMemory = nullptr;
  vk::raii::Buffer indexBuffer = nullptr;
  vk::raii::DeviceMemory indexBufferMemory = nullptr;

  std::vector<vk::raii::Buffer> uniformBuffers;
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
  std::vector<void*> uniformBuffersMapped;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;

  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;

  std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
  std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
  std::vector<vk::raii::Fence> inFlightFences;
  uint32_t frameIndex = 0;

  bool framebufferResized = false;

  std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};

  void initWindow() {
    window = Window("Vulkan", WIDTH, HEIGHT);
    window.setUserPointer(this);
    window.setFramebuffferSizeCallback(framebufferResizeCallback);
  }

  static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
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
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createTextureImage();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
  }

  void createTextureImage() {}

  void createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layout(FRAMES_IN_FLIGHT, *descriptorSetLayout);
    vk::DescriptorSetAllocateInfo descriptorSetAI{
        .descriptorPool = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layout.size()),
        .pSetLayouts = layout.data(),
    };

    descriptorSets.clear();
    descriptorSets = device.allocateDescriptorSets(descriptorSetAI);

    for (auto i = 0; i < descriptorSets.size(); i++) {
      vk::DescriptorBufferInfo descriptorBufferInfo{
          .buffer = uniformBuffers[i],
          .offset = 0,
          .range = sizeof(ShaderData),
      };

      vk::WriteDescriptorSet writeDescriptorSet{
          .dstSet = descriptorSets[i],
          .dstBinding = 0,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eUniformBuffer,
          .pBufferInfo = &descriptorBufferInfo,
      };

      device.updateDescriptorSets(writeDescriptorSet, {});
    }
  }

  void createDescriptorPool() {
    vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = FRAMES_IN_FLIGHT,
    };

    vk::DescriptorPoolCreateInfo poolCI{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = FRAMES_IN_FLIGHT,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };

    descriptorPool = vk::raii::DescriptorPool(device, poolCI);
  }

  void createUniformBuffers() {
    for (auto i = 0; i < FRAMES_IN_FLIGHT; i++) {
      vk::DeviceSize bufferSize = sizeof(ShaderData);
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

  void createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                                    vk::ShaderStageFlagBits::eVertex, nullptr);
    vk::DescriptorSetLayoutCreateInfo uboLayoutCI{
        .bindingCount = 1,
        .pBindings = &uboLayoutBinding,
    };

    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, uboLayoutCI);
  }

  void updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    ShaderData ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.projection = glm::perspective(
        glm::radians(45.0f), static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height),
        0.1f, 10.0f);
    ubo.projection[1][1] *= -1;

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
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

  void createVertexBuffer() {
    const auto bufferSize = sizeof(Vertex) * vertices.size();
    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingBufferMemory = nullptr;
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer,
                 stagingBufferMemory);
    void* data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, vertices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
  }

  void createIndexBuffer() {
    const auto bufferSize = sizeof(indices[0]) * indices.size();

    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingBufferMemory = nullptr;

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer,
                 stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, indices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);
  }

  void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
    vk::CommandBufferAllocateInfo commandBufferAI{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    vk::raii::CommandBuffer copyCommandBuffer = std::move(device.allocateCommandBuffers(commandBufferAI).front());

    copyCommandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    copyCommandBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
    copyCommandBuffer.end();

    queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*copyCommandBuffer});
    queue.waitIdle();
  }

  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    const auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type");
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window.handle)) {
      glfwPollEvents();
      drawFrame();
    }

    device.waitIdle();
  }

  void recreateSwapchain() {
    int width, height;
    glfwGetFramebufferSize(window.handle, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window.handle, &width, &height);
      glfwWaitEvents();
    }
    device.waitIdle();

    cleanupSwapchain();

    createSwapchain();
    createImageViews();
  }

  void cleanupSwapchain() {
    swapchainImageViews.clear();
    swapchain = nullptr;
  }

  void drawFrame() {
    auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
    auto [result, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapchain();
      return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
      assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
      throw std::runtime_error("failed to aquire next swapchain image");
    }
    device.resetFences(*inFlightFences[frameIndex]);

    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitDestinationStageMask{vk::PipelineStageFlagBits::eColorAttachmentOutput};
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
        .pWaitDstStageMask = &waitDestinationStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[frameIndex],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex],
    };

    updateUniformBuffer(frameIndex);

    queue.submit(submitInfo, *inFlightFences[frameIndex]);

    result = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for fence");
    }

    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapchain,
        .pImageIndices = &imageIndex,
    };

    result = queue.presentKHR(presentInfo);
    if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR || framebufferResized) {
      framebufferResized = false;
      recreateSwapchain();
    }

    frameIndex = (frameIndex + 1) % FRAMES_IN_FLIGHT;
  }

  void cleanup() {
    cleanupSwapchain();
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
    const auto unsupportedLayerIt = std::ranges::find_if(requiredLayers, [&layerProperties](const auto&
    requiredLayer) {
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
    if (glfwCreateWindowSurface(*instance, window.handle, nullptr, &_surface) != VK_SUCCESS) {
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

    const auto deviceIt = std::ranges::find_if(physicalDevices, [&](const auto& pd) { return isDeviceSuitable(pd);
    }); if (deviceIt == physicalDevices.end()) {
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
    if (std::ranges::any_of(requiredDeviceExtension, [&availableDeviceExtensions](const auto&
    requiredDeviceExtension) {
          return std::ranges::none_of(availableDeviceExtensions,
                                      [requiredDeviceExtension](const auto& availableExtension) {
                                        return strcmp(availableExtension.extensionName, requiredDeviceExtension) ==
                                        0;
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
          features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2)) {
      return false;
    }

    return true;
  }

  void createLogicalDevice() {
    const std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

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
            {.synchronization2 = true, .dynamicRendering = true},
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
    queue = vk::raii::Queue(device, queueIndex, 0);
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
    assert(std::ranges::any_of(availablePresentModes, [](const auto& mode) { return mode ==
    vk::PresentModeKHR::eFifo; }));

    if (std::ranges::any_of(availablePresentModes, [](const auto& mode) { return mode ==
    vk::PresentModeKHR::eMailbox; })) {
      return vk::PresentModeKHR::eMailbox;
    }

    return vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window.handle, &width, &height);

    return {
        std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
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

    const auto vertexBindingDescription = Vertex::getBindingDescription();
    const auto vertexAttributeDescriptions = Vertex::getAttributeDescription();
    vk::PipelineVertexInputStateCreateInfo vertexInputCI{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertexBindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size()),
        .pVertexAttributeDescriptions = vertexAttributeDescriptions.data(),
    };
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
        .frontFace = vk::FrontFace::eCounterClockwise,
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
        .setLayoutCount = 1,
        .pSetLayouts = &*descriptorSetLayout,
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

  void createRenderPass() {
    vk::SubpassDependency subpassDependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = {},
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .srcAccessMask = {},
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
    };

    vk::RenderPassCreateInfo renderPassCI{
        .dependencyCount = 1,
        .pDependencies = &subpassDependency,
    };
  }

  void createCommandPool() {
    vk::CommandPoolCreateInfo commandPoolCI{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueIndex,
    };

    commandPool = vk::raii::CommandPool(device, commandPoolCI);
  }

  void createCommandBuffers() {
    vk::CommandBufferAllocateInfo commandBufferAI{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = FRAMES_IN_FLIGHT,
    };

    commandBuffers = vk::raii::CommandBuffers(device, commandBufferAI);
  }

  void recordCommandBuffer(uint32_t imageIndex) {
    commandBuffers[frameIndex].begin({});

    transitionImageLayout(imageIndex, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
                          vk::AccessFlagBits2::eColorAttachmentWrite,
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    vk::ClearValue clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::RenderingAttachmentInfo renderingAttachmentInfo{
        .imageView = swapchainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearValue,
    };

    vk::RenderingInfo renderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = swapchainExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &renderingAttachmentInfo,
    };

    commandBuffers[frameIndex].beginRendering(renderingInfo);

    commandBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
    commandBuffers[frameIndex].bindVertexBuffers(0, *vertexBuffer, {0});
    commandBuffers[frameIndex].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
    commandBuffers[frameIndex].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
                                                  *descriptorSets[frameIndex], nullptr);
    commandBuffers[frameIndex].setViewport(0, vk::Viewport{0.0f, 0.0f, static_cast<float>(swapchainExtent.width),
                                                           static_cast<float>(swapchainExtent.height)});
    commandBuffers[frameIndex].setScissor(0, vk::Rect2D{vk::Offset2D{0, 0}, swapchainExtent});

    commandBuffers[frameIndex].drawIndexed(indices.size(), 1, 0, 0, 0);

    commandBuffers[frameIndex].endRendering();

    transitionImageLayout(imageIndex, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                          vk::AccessFlagBits2::eColorAttachmentWrite, {},
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits2::eBottomOfPipe);

    commandBuffers[frameIndex].end();
  }

  void transitionImageLayout(uint32_t imageIndex, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                             vk::AccessFlags2 srcAccessMask, vk::AccessFlags2 dstAccessMask,
                             vk::PipelineStageFlags2 srcStageMask, vk::PipelineStageFlags2 dstStageMask) {
    vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = srcStageMask,
        .srcAccessMask = srcAccessMask,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = swapchainImages[imageIndex],
        .subresourceRange =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };

    vk::DependencyInfo dependencyInfo{
        .dependencyFlags = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };

    commandBuffers[frameIndex].pipelineBarrier2(dependencyInfo);
  }

  [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& shaderCode) const {
    vk::ShaderModuleCreateInfo shaderModuleCI{
        .codeSize = shaderCode.size(),
        .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data()),
    };

    vk::raii::ShaderModule shaderModule{device, shaderModuleCI};

    return shaderModule;
  }

  void createSyncObjects() {
    assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

    for (auto i = 0; i < swapchainImages.size(); i++) {
      renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }

    for (auto i = 0; i < FRAMES_IN_FLIGHT; i++) {
      presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
      inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
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
  if (!glfwInit()) {
    std::cerr << "failed to initialize glfw";
    return EXIT_FAILURE;
  }

  try {
    HelloTriangleApplication app;
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    glfwTerminate();
    return EXIT_FAILURE;
  }

  glfwTerminate();

  return EXIT_SUCCESS;
}
