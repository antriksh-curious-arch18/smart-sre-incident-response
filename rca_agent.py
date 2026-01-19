import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class IncidentResponseAgent:
    """
    AI-Driven Incident Response Agent utilizing TF-IDF (Term Frequency-Inverse Document Frequency)
    and Cosine Similarity to perform semantic searches on historical log data.
    
    Attributes:
        data (list): A list of historical incident dictionaries.
        vectorizer (TfidfVectorizer): Scikit-learn vectorizer model.
        tfidf_matrix (sparse matrix): The vectorized representation of the knowledge base.
    """

    def __init__(self, data_source):
        """
        Initializes the agent with data and trains the vector model.
        """
        self.data = data_source
        self.vectorizer = None
        self.tfidf_matrix = None
        self._train_model()

    def _train_model(self):
        """
        Internal method to vectorize the knowledge base (Corpus).
        Concatenates Error + Root Cause + Fix to create a rich context for matching.
        """
        print("⚙️  Initializing AI Agent...")
        print("🧠  Vectorizing Knowledge Base...")
        
        # Create a unified corpus string for each incident
        corpus = [
            f"{item['ErrorMessage']} {item['RootCause']} {item['FixApplied']}" 
            for item in self.data
        ]
        
        # Initialize and fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        print(f"✅  Model Trained successfully on {len(self.data)} historical incidents.\n")

    def search(self, query, threshold=0.1):
        """
        Performs a semantic search for the user query.
        
        Args:
            query (str): The incident description or error log.
            threshold (float): Minimum confidence score required to return a match.
            
        Returns:
            dict: The best matching incident and its metadata, or None if no match found.
        """
        # Convert user query to vector
        query_vec = self.vectorizer.transform([query])
        
        # Calculate Cosine Similarity against all historical records
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Find the index of the highest score
        best_match_index = np.argmax(similarities)
        confidence_score = similarities[best_match_index]
        
        if confidence_score < threshold:
            return None
        
        return {
            "incident": self.data[best_match_index],
            "score": confidence_score
        }

    def analyze_impact(self, incident_data, score):
        """
        Applies heuristic logic to determine Severity, Team Assignment, and Cost Impact.
        This mimics 'Managerial Decision Making' based on keywords.
        """
        text_content = (incident_data['ErrorMessage'] + " " + incident_data['RootCause']).lower()
        
        # 1. Team Assignment Logic
        assigned_team = "General Support"
        if any(x in text_content for x in ['database', 'sql', 'postgres', 'mongo', 'redis', 'db']):
            assigned_team = "🗄️ Database Operations"
        elif any(x in text_content for x in ['pod', 'container', 'k8s', 'kubernetes', 'docker', 'node']):
            assigned_team = "☸️ Platform Engineering / SRE"
        elif any(x in text_content for x in ['network', 'timeout', 'dns', 'port', 'connection', '503', '504']):
            assigned_team = "🌐 Network Operations"
        elif any(x in text_content for x in ['java', 'python', 'npm', 'code', 'exception', 'nullpointer']):
            assigned_team = "💻 Application Development"

        # 2. Severity Classification Logic
        severity = "🟡 Medium (P3)"
        if any(x in text_content for x in ['fatal', 'crash', 'down', 'deadlock', 'oomkilled', 'security', 'attack']):
            severity = "🔴 CRITICAL (P1)"
        elif any(x in text_content for x in ['warning', 'deprecated', 'slow']):
            severity = "🟢 Low (P4)"

        # 3. Cost Estimation Logic (Simulated Business Impact)
        cost_impact = "$100/hr"
        if "CRITICAL" in severity:
            cost_impact = "$30,000/hr (High Risk)"
        elif "Medium" in severity:
            cost_impact = "$5,000/hr"

        return {
            "team": assigned_team,
            "severity": severity,
            "cost": cost_impact
        }

# ==========================================
# DATASET: 50 Simulated DevOps Incidents
# ==========================================
KNOWLEDGE_BASE = [
  {
    "IncidentID": "INC-1001",
    "Date": "2023-10-01",
    "ErrorMessage": "FATAL: CreateIndex failed. Disk is full.",
    "RootCause": "PostgreSQL transaction log (WAL) partition reached 100% capacity due to unexpected spike in write operations.",
    "FixApplied": "Expanded EBS volume size by 200GB and configured log rotation policy."
  },
  {
    "IncidentID": "INC-1002",
    "Date": "2023-10-02",
    "ErrorMessage": "503 Service Unavailable (Upstream Request Timeout)",
    "RootCause": "Nginx ingress controller overloaded; worker connections exhausted.",
    "FixApplied": "Increased worker_connections in Nginx config and scaled ingress replicas from 3 to 5."
  },
  {
    "IncidentID": "INC-1003",
    "Date": "2023-10-03",
    "ErrorMessage": "OOMKilled: Container 'payment-service' used more than 512Mi memory.",
    "RootCause": "Memory leak in the payment processing module processing large CSV batches.",
    "FixApplied": "Increased memory limit to 1Gi and rolled out hotfix limiting batch size."
  },
  {
    "IncidentID": "INC-1004",
    "Date": "2023-10-05",
    "ErrorMessage": "Error: x509: certificate has expired or is not yet valid.",
    "RootCause": "Internal TLS certificate for auth-service expired; auto-renewal cron job failed silently.",
    "FixApplied": "Manually renewed certificate and added alerting for failed Cert-Manager jobs."
  },
  {
    "IncidentID": "INC-1005",
    "Date": "2023-10-07",
    "ErrorMessage": "Deadlock found when trying to get lock; try restarting transaction.",
    "RootCause": "Concurrent updates to the 'inventory' table during peak traffic caused race conditions.",
    "FixApplied": "Optimized SQL query ordering and implemented retry logic in the backend application."
  },
  {
    "IncidentID": "INC-1006",
    "Date": "2023-10-08",
    "ErrorMessage": "CrashLoopBackOff: Back-off restarting failed container.",
    "RootCause": "Application panic on startup due to missing environment variable 'DB_HOST'.",
    "FixApplied": "Updated Kubernetes ConfigMap to include missing variable and redeployed."
  },
  {
    "IncidentID": "INC-1007",
    "Date": "2023-10-10",
    "ErrorMessage": "Redis::CannotConnectError: Connection refused.",
    "RootCause": "Redis cluster leader election took too long during a node failover.",
    "FixApplied": "Adjusted Redis cluster timeout settings and upgraded instance type for stability."
  },
  {
    "IncidentID": "INC-1008",
    "Date": "2023-10-12",
    "ErrorMessage": "java.lang.OutOfMemoryError: Java heap space",
    "RootCause": "JVM Heap size insufficient for new reporting microservice workload.",
    "FixApplied": "Tuned JVM flags to -Xmx2048m and optimized garbage collection strategy."
  },
  {
    "IncidentID": "INC-1009",
    "Date": "2023-10-15",
    "ErrorMessage": "Connection timed out: connect(2)",
    "RootCause": "Security Group rule change accidentally blocked port 8080 traffic between app and cache subnets.",
    "FixApplied": "Reverted Security Group changes via Terraform to allow internal traffic on port 8080."
  },
  {
    "IncidentID": "INC-1010",
    "Date": "2023-10-18",
    "ErrorMessage": "ImagePullBackOff: rpc error: code = Unknown desc = Error response from daemon",
    "RootCause": "Docker registry credentials expired in the Kubernetes secret.",
    "FixApplied": "Rotated registry service principal password and updated the K8s pull secret."
  },
  {
    "IncidentID": "INC-1011",
    "Date": "2023-10-20",
    "ErrorMessage": "Too many open files",
    "RootCause": "File descriptor limit reached on the API gateway node.",
    "FixApplied": "Increased 'ulimit -n' in system configuration and restarted the service."
  },
  {
    "IncidentID": "INC-1012",
    "Date": "2023-10-22",
    "ErrorMessage": "DNS_PROBE_FINISHED_NXDOMAIN",
    "RootCause": "CoreDNS pods were evicted due to node pressure, causing internal DNS failures.",
    "FixApplied": "Added PodDisruptionBudget for CoreDNS and scaled cluster autoscaler."
  },
  {
    "IncidentID": "INC-1013",
    "Date": "2023-10-25",
    "ErrorMessage": "CPU Throttling detected: 99% usage.",
    "RootCause": "Crypto-mining malware detected on a compromised dev instance.",
    "FixApplied": "Terminated compromised instance, rotated keys, and tightened SSH access rules."
  },
  {
    "IncidentID": "INC-1014",
    "Date": "2023-10-28",
    "ErrorMessage": "429 Too Many Requests",
    "RootCause": "DDoS attack targeting the login endpoint.",
    "FixApplied": "Enabled WAF rate limiting and blocked offending IP subnets."
  },
  {
    "IncidentID": "INC-1015",
    "Date": "2023-10-30",
    "ErrorMessage": "IntegrityConstraintViolation: Duplicate entry",
    "RootCause": "UUID collision in the user-generation logic.",
    "FixApplied": "Switched from v1 UUIDs to v4 random UUIDs to ensure uniqueness."
  },
  {
    "IncidentID": "INC-1016",
    "Date": "2023-11-01",
    "ErrorMessage": "Kafka: NOT_LEADER_FOR_PARTITION",
    "RootCause": "Kafka broker unbalance; partition leader was offline.",
    "FixApplied": "Triggered a controlled Kafka partition rebalance using cruise-control."
  },
  {
    "IncidentID": "INC-1017",
    "Date": "2023-11-03",
    "ErrorMessage": "Elasticsearch status: Red",
    "RootCause": "Unassigned shards due to failure of two data nodes simultaneously.",
    "FixApplied": "Provisioned new data nodes and initiated shard recovery."
  },
  {
    "IncidentID": "INC-1018",
    "Date": "2023-11-05",
    "ErrorMessage": "SocketTimeoutException: Read timed out",
    "RootCause": "Third-party payment gateway API experienced high latency.",
    "FixApplied": "Increased client-side timeout to 10s and implemented circuit breaker pattern."
  },
  {
    "IncidentID": "INC-1019",
    "Date": "2023-11-08",
    "ErrorMessage": "npm ERR! code E404",
    "RootCause": "Private npm package was accidentally deleted from the artifact registry.",
    "FixApplied": "Restored package from backup and locked registry deletion permissions."
  },
  {
    "IncidentID": "INC-1020",
    "Date": "2023-11-10",
    "ErrorMessage": "Terraform Error: Error acquiring the state lock",
    "RootCause": "Previous CI/CD pipeline was cancelled abruptly, leaving the DynamoDB lock active.",
    "FixApplied": "Manually released the Terraform state lock using 'force-unlock'."
  },
  {
    "IncidentID": "INC-1021",
    "Date": "2023-11-12",
    "ErrorMessage": "S3: AccessDenied",
    "RootCause": "IAM policy change inadvertently removed 's3:GetObject' from the web-app role.",
    "FixApplied": "Restored correct IAM policy permissions via CloudFormation update."
  },
  {
    "IncidentID": "INC-1022",
    "Date": "2023-11-15",
    "ErrorMessage": "Jenkins: No space left on device",
    "RootCause": "Build artifacts from stale branches consumed all disk space on the build agent.",
    "FixApplied": "Ran cleanup script to delete artifacts older than 30 days and increased disk size."
  },
  {
    "IncidentID": "INC-1023",
    "Date": "2023-11-17",
    "ErrorMessage": "ModuleNotFoundError: No module named 'requests'",
    "RootCause": "requirements.txt was not updated after a code merge, causing build failure.",
    "FixApplied": "Added missing dependency to requirements.txt and rebuilt the container image."
  },
  {
    "IncidentID": "INC-1024",
    "Date": "2023-11-19",
    "ErrorMessage": "Node NotReady",
    "RootCause": "Kubelet stopped posting heartbeats due to high system load averages.",
    "FixApplied": "Drained the node, rebooted, and cordoned it for investigation."
  },
  {
    "IncidentID": "INC-1025",
    "Date": "2023-11-22",
    "ErrorMessage": "FATAL: password authentication failed for user 'app_user'",
    "RootCause": "Database password rotation occurred but the app secret wasn't updated.",
    "FixApplied": "Updated the Kubernetes secret with the new DB password and restarted pods."
  },
  {
    "IncidentID": "INC-1026",
    "Date": "2023-11-24",
    "ErrorMessage": "502 Bad Gateway",
    "RootCause": "Backend gRPC service crashed due to unhandled exception.",
    "FixApplied": "Patched the exception handling logic and redeployed the gRPC service."
  },
  {
    "IncidentID": "INC-1027",
    "Date": "2023-11-26",
    "ErrorMessage": "HSTS: The connection to the site is not secure",
    "RootCause": "Subdomain was missing from the HSTS preload list and certificate SAN.",
    "FixApplied": "Issued new wildcard certificate covering the subdomain."
  },
  {
    "IncidentID": "INC-1028",
    "Date": "2023-11-28",
    "ErrorMessage": "Error: ENOSPC: no space left on device, write",
    "RootCause": "Log files in /var/log/docker filled the root partition.",
    "FixApplied": "Truncated logs and configured Docker daemon to limit log file size."
  },
  {
    "IncidentID": "INC-1029",
    "Date": "2023-11-30",
    "ErrorMessage": "ActiveMQ: Usage Manager Memory Limit Reached",
    "RootCause": "Consumer service stalled, causing message queue buildup.",
    "FixApplied": "Restarted consumer service and purged dead-letter queue."
  },
  {
    "IncidentID": "INC-1030",
    "Date": "2023-12-02",
    "ErrorMessage": "Error: ElementClickInterceptedException",
    "RootCause": "UI Selenium tests failed due to a new popup modal blocking buttons.",
    "FixApplied": "Updated test scripts to close the modal before clicking."
  },
  {
    "IncidentID": "INC-1031",
    "Date": "2023-12-04",
    "ErrorMessage": "Failed to mount volume: specific path not found",
    "RootCause": "NFS server IP changed but the PV definition was static.",
    "FixApplied": "Updated PV definition with the new NFS DNS name instead of hardcoded IP."
  },
  {
    "IncidentID": "INC-1032",
    "Date": "2023-12-06",
    "ErrorMessage": "Metric query returned empty result",
    "RootCause": "Prometheus scraper configuration regex did not match new pod naming convention.",
    "FixApplied": "Updated Prometheus ServiceMonitor regex to include new pod names."
  },
  {
    "IncidentID": "INC-1033",
    "Date": "2023-12-08",
    "ErrorMessage": "Git: RPC failed; curl 56 OpenSSL SSL_read: Connection was reset",
    "RootCause": "Git repository size too large for standard buffer settings during clone.",
    "FixApplied": "Increased 'http.postBuffer' in git config on the CI runner."
  },
  {
    "IncidentID": "INC-1034",
    "Date": "2023-12-10",
    "ErrorMessage": "Lambda ThrottlingException",
    "RootCause": "Concurrent Lambda executions exceeded the regional account limit.",
    "FixApplied": "Requested AWS quota increase and optimized Lambda trigger batch size."
  },
  {
    "IncidentID": "INC-1035",
    "Date": "2023-12-12",
    "ErrorMessage": "Segmentation fault (core dumped)",
    "RootCause": "Buffer overflow in a legacy C++ component processing image data.",
    "FixApplied": "Applied patch to validate input buffer length before processing."
  },
  {
    "IncidentID": "INC-1036",
    "Date": "2023-12-15",
    "ErrorMessage": "ZooKeeper: ConnectionLossException",
    "RootCause": "Network partition between ZooKeeper quorum nodes.",
    "FixApplied": "Restored network connectivity and verified quorum sync status."
  },
  {
    "IncidentID": "INC-1037",
    "Date": "2023-12-18",
    "ErrorMessage": "Error: listen EADDRINUSE: address already in use :::3000",
    "RootCause": "Developer ran two instances of the app on the same port in the dev environment.",
    "FixApplied": "Killed the zombie process occupying port 3000."
  },
  {
    "IncidentID": "INC-1038",
    "Date": "2023-12-20",
    "ErrorMessage": "Ansible: unreachable: Failed to connect to the host via ssh",
    "RootCause": "SSH key fingerprint changed on target servers after OS rebuild.",
    "FixApplied": "Cleared known_hosts on the Ansible control node and re-accepted keys."
  },
  {
    "IncidentID": "INC-1039",
    "Date": "2023-12-22",
    "ErrorMessage": "CloudFormation: Stack CREATE_FAILED",
    "RootCause": "Circular dependency detected between Load Balancer and Security Group resources.",
    "FixApplied": "Refactored CloudFormation template to decouple resources."
  },
  {
    "IncidentID": "INC-1040",
    "Date": "2023-12-24",
    "ErrorMessage": "504 Gateway Time-out",
    "RootCause": "Slow database migration script locked tables longer than the load balancer timeout.",
    "FixApplied": "Scheduled migration for maintenance window and used non-locking schema changes."
  },
  {
    "IncidentID": "INC-1041",
    "Date": "2023-12-26",
    "ErrorMessage": "Vault: sealed",
    "RootCause": "Vault server restarted unexpectedly and did not auto-unseal.",
    "FixApplied": "Manually unsealed Vault using the shamir key shards."
  },
  {
    "IncidentID": "INC-1042",
    "Date": "2023-12-28",
    "ErrorMessage": "Error: self signed certificate in certificate chain",
    "RootCause": "NodeJS app rejected the internal CA certificate during API call.",
    "FixApplied": "Set NODE_EXTRA_CA_CERTS environment variable to include the internal CA."
  },
  {
    "IncidentID": "INC-1043",
    "Date": "2023-12-30",
    "ErrorMessage": "MongoTimeoutError: Server selection timed out",
    "RootCause": "MongoDB primary node stepped down and secondary took too long to elect.",
    "FixApplied": "Adjusted election timeout settings and verified heartbeat frequency."
  },
  {
    "IncidentID": "INC-1044",
    "Date": "2024-01-02",
    "ErrorMessage": "Python: ImportError: cannot import name 'soft_unicode'",
    "RootCause": "Incompatible version of 'MarkupSafe' library installed by a sub-dependency.",
    "FixApplied": "Pinned 'MarkupSafe' version to 2.0.1 in setup.py."
  },
  {
    "IncidentID": "INC-1045",
    "Date": "2024-01-04",
    "ErrorMessage": "Azure: StorageQuotaExceeded",
    "RootCause": "Azure Blob Storage container reached its configured size limit.",
    "FixApplied": "Increased quota limit and implemented lifecycle policy to archive old blobs."
  },
  {
    "IncidentID": "INC-1046",
    "Date": "2024-01-06",
    "ErrorMessage": "Liquibase: CheckSum Check Failed",
    "RootCause": "A previously run migration file was manually modified in the codebase.",
    "FixApplied": "Reverted the manual change and created a new changeset for the fix."
  },
  {
    "IncidentID": "INC-1047",
    "Date": "2024-01-08",
    "ErrorMessage": "Error: Pod has unbound immediate PersistentVolumeClaims",
    "RootCause": "Requested storage class 'fast-ssd' did not exist in the new cluster.",
    "FixApplied": "Created the missing StorageClass and re-applied the PVC."
  },
  {
    "IncidentID": "INC-1048",
    "Date": "2024-01-10",
    "ErrorMessage": "Datadog: Agent is not reporting",
    "RootCause": "Datadog API key was invalid in the agent configuration.",
    "FixApplied": "Updated the Datadog API key in the Helm chart values."
  },
  {
    "IncidentID": "INC-1049",
    "Date": "2024-01-12",
    "ErrorMessage": "CORS policy: No 'Access-Control-Allow-Origin' header is present",
    "RootCause": "Frontend domain changed but the backend CORS whitelist wasn't updated.",
    "FixApplied": "Added the new frontend URL to the backend CORS configuration."
  },
  {
    "IncidentID": "INC-1050",
    "Date": "2024-01-15",
    "ErrorMessage": "Argocd: Application OutOfSync",
    "RootCause": "Manual kubectl edit made to a deployment, causing drift from Git state.",
    "FixApplied": "Synced ArgoCD to overwrite manual changes and restore state from Git."
  }
]

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    # Initialize the Agent
    agent = IncidentResponseAgent(KNOWLEDGE_BASE)
    
    print("🧪  STARTING DIAGNOSTIC TEST SUITE...\n")
    
    # Test Scenarios
    scenarios = [
        "The database disk is completely full causing crashes",
        "Application is failing with Out of Memory errors",
        "API calls are timing out and service is unavailable",
        "The printer is jammed" # Control case (should find nothing)
    ]
    
    for query in scenarios:
        print(f"🔎  Query: '{query}'")
        
        # 1. Search
        result = agent.search(query)
        
        if result:
            # 2. Analyze
            analysis = agent.analyze_impact(result['incident'], result['score'])
            
            # 3. Report
            print(f"✅  MATCH FOUND (Confidence: {round(result['score']*100, 2)}%)")
            print(f"    • Incident ID:  {result['incident']['IncidentID']}")
            print(f"    • Root Cause:   {result['incident']['RootCause']}")
            print(f"    • Suggested Fix:{result['incident']['FixApplied']}")
            print(f"    --- Managerial Insights ---")
            print(f"    • Severity:     {analysis['severity']}")
            print(f"    • Team Routing: {analysis['team']}")
            print(f"    • Risk Impact:  {analysis['cost']}")
        else:
            print("❌  No relevant historical data found.")
            
        print("-" * 60 + "\n")