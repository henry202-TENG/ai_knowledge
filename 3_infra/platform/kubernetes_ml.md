# Kubernetes for ML

使用 Kubernetes 部署和管理 ML 工作負載的技術與最佳實踐。

---

## 1. 什麼是？

### 簡單範例

```
傳統部署:
  手動管理服務器 → 安裝依賴 → 部署模型
  (繁瑣、難擴展)

Kubernetes:
  聲明式配置 → 自動調度 → 自我修復
  (自動擴展、容錯、高可用)
```

---

## 2. 核心概念

### Pod 設計

```yaml
# ml-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-inference
spec:
  containers:
  - name: vllm
    image: vllm/vllm:latest
    resources:
      requests:
        nvidia.com/gpu: 1
        memory: "32Gi"
        cpu: "8"
      limits:
        nvidia.com/gpu: 1
        memory: "40Gi"
        cpu: "16"
    env:
    - name: MODEL_NAME
      value: "meta-llama/Llama-2-7b"
    volumeMounts:
    - name: model-cache
      mountPath: /models
  volumes:
  - name: model-cache
    persistentVolumeClaim:
      claimName: model-storage
```

### GPU 調度

```yaml
# gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-training
spec:
  replicas: 2
  template:
    spec:
      nodeSelector:
        gpu-type: "A100"
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: trainer
        image: pytorch/torchrun:latest
        resources:
          limits:
            nvidia.com/gpu: 2
```

---

## 3. ML 工作負載

### Training Job

```yaml
# pytorch-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-training
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0-cuda11.7
        command:
        - torchrun
        - --nproc_per_node=8
        - train.py
        env:
        - name: WORLD_SIZE
          value: "8"
        - name: MASTER_ADDR
          value: "ml-headless-service"
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: "256Gi"
```

### Distributed Training

```yaml
# headless-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-headless-service
spec:
  clusterIP: None
  selector:
    role: trainer
  ports:
  - name: grpc
    port: 29500
  - name: nccl
    port: 29501
```

### Inference Service

```yaml
# inference-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-inference
spec:
  selector:
    app: llm-server
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

---

## 4. 存儲管理

### Model Storage

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: nfs-storage
---
# StorageClass for fast NVMe
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nvme-storage
provisioner: pd.csi.storage.gke.io
volumeBindingMode: WaitForFirstConsumer
parameters:
  type: pd-ssd
```

### 數據集管理

```python
from kubernetes import client

class MLStorageManager:
    """ML 存儲管理器"""

    def __init__(self):
        self.core = client.CoreV1Api()

    def create_dataset_pvc(self, name, size, storage_class):
        """創建數據集 PVC"""

        pvc = client.V1PersistentVolumeClaim(
            metadata=client.V1ObjectMeta(name=name),
            spec=client.V1PersistentVolumeClaimSpec(
                access_modes=["ReadOnlyMany"],
                resources=client.V1ResourceRequirements(
                    requests={"storage": size}
                ),
                storage_class_name=storage_class
            )
        )

        return self.core.create_namespaced_persistent_volume_claim(
            namespace="ml",
            body=pvc
        )
```

---

## 5. 網路配置

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  rules:
  - host: llm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llm-inference
            port:
              number: 8000
```

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-training-policy
spec:
  podSelector:
    matchLabels:
      role: trainer
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: scheduler
    ports:
    - protocol: TCP
      port: 29500
  egress:
  - to:
    - podSelector:
        matchLabels:
          role: trainer
    ports:
    - protocol: TCP
      port: 29500
```

---

## 6. 自動擴展

### KEDA 事件驅動

```yaml
# keda-scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-scaler
spec:
  scaleTargetRef:
    name: llm-server
  pollingInterval: 15
  cooldownPeriod: 300
  minReplicaCount: 2
  maxReplicaCount: 20
  triggers:
  - type: cpu
    metricType: Utilization
    metadata:
      value: "70"
  - type: memory
    metricType: Utilization
    metadata:
      value: "80"
  - type: kafka
    metadata:
      bootstrapServers: kafka:9092
      consumerGroup: llm-group
      topic: llm-requests
      lagThreshold: "100"
```

### GPU 擴展

```python
class GPUScaler:
    """GPU 資源管理器"""

    def __init__(self, k8s_client):
        self.client = k8s_client

    def get_available_gpu(self):
        """獲取可用 GPU 數量"""

        nodes = self.client.list_node()

        total = 0
        for node in nodes.items:
            gpu = node.status.capacity.get("nvidia.com/gpu", "0")
            total += int(gpu)

        return total

    def scale_training(self, replicas):
        """擴展訓練任務"""

        # 使用 Deployment scale
        self.client.patch_namespaced_deployment_scale(
            name="ml-training",
            namespace="ml",
            body={"spec": {"replicas": replicas}}
        )
```

---

## 7. 監控

### GPU 監控

```yaml
# dcgm-exporter.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    spec:
      containers:
      - name: exporter
        image: nvcr.io/nvidia/dcgm-exporter:latest
        env:
        - name: DCGM_EXPORTER_GPU_METRICS
          value: "1536,1537,1540,1541,1542"
        ports:
        - containerPort: 9400
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Metrics Dashboard

```python
# 關鍵監控指標
ML_METRICS = {
    # 訓練指標
    "training_loss": "訓練損失",
    "gpu_utilization": "GPU 利用率",
    "gpu_memory": "GPU 記憶體使用",
    "training_throughput": "訓練吞吐量",

    # 推理指標
    "inference_latency": "推理延遲",
    "requests_per_second": "每秒請求數",
    "model_loaded": "模型加載狀態",
    "queue_length": "請求隊列長度",

    # 集群指標
    "node_cpu": "節點 CPU",
    "node_memory": "節點記憶體",
    "pod_restarts": "Pod 重啟次數"
}
```

---

## 8. 最佳實踐

### 資源配置

```yaml
# resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-quota
spec:
  hard:
    requests.cpu: "64"
    requests.memory: "256Gi"
    limits.cpu: "128"
    limits.memory: "512Gi"
    nvidia.com/gpu: "8"
    pods: "20"
```

### 故障排除

```
常見問題:

1. GPU OOM
   解決: 減少 batch size、启用 swap、使用更小的模型

2. Pod 調度失敗
   解決: 檢查資源請求、節點標籤、污點容忍

3. 網路連接問題
   解決: 檢查 NetworkPolicy、Service、DNS 配置

4. 訓練卡住
  解決: 檢查 NCCL 配置、超時設置、日誌
```

---

## 9. 與相關技術

| 技術 | 關係 |
|------|------|
| **Model Serving** | K8s 部署推理服務 |
| **Helm** | 部署 Charts |
| **Istio** | 服務網格 |
| **Prometheus** | 監控指標 |

---

## 延伸閱讀

- [Kubernetes ML Guide](https://kubernetes.io/docs/tasks/job/)
- [KubeFlow](https://www.kubeflow.org/)
- [KEDA](https://keda.sh/)