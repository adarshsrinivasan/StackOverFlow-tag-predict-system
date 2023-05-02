pipeline {
    agent any

    environment {
        PROJECT_ID = "your_project_id"
        BUCKET_NAME = "your_bucket_name"
        CLUSTER_NAME = "your_cluster_name"
        REGION = "your_region"
        DATASET = "your_dataset_name"
        TABLE = "your_table_name"
        SCRIPTS_DIR = "scripts"
        VENV_DIR = "venv"
        IMAGE_NAME = "your_image_name"
        TAG_NAME = "your_tag_name"
        DEPLOYMENT_NAME = "your_deployment_name"
    }

    stages {
        stage("Setup") {
            steps {
                withCredentials([usernamePassword(credentialsId: 'git-cred', usernameVariable: 'GIT_USERNAME', passwordVariable: 'GIT_PASSWORD')]) {
                    sh "git config --global credential.helper 'store --file ~/.git-credentials'"
                    sh "echo -e 'protocol=https\nhost=github.com\nusername=$GIT_USERNAME\npassword=$GIT_PASSWORD' >> ~/.git-credentials"
                    sh "rm -rf StackOverFlow-tag-predict-system"
                    sh "git clone https://github.com/adarshsrinivasan/StackOverFlow-tag-predict-system.git"
                    sh "gcloud container clusters get-credentials jenkins-cd  --region us-central1-b"
                    sh "kubectl get nodes"
                }
            }
        }
        stage("Filter and Store Data") {
            steps {
                sh "kubectl delete job bdaproject-stage1-job || true"
                sh "kubectl apply -f ./StackOverFlow-tag-predict-system/deployment/stage1-job.yaml"
                sh "kubectl wait --for=condition=complete --timeout=10m job/bdaproject-stage1-job"
            }
        }
        stage("Train model and upload") {
            steps {
                sh "kubectl delete job bdaproject-stage2-job || true"
                sh "kubectl apply -f ./StackOverFlow-tag-predict-system/deployment/stage2-job.yaml"
                sh "kubectl wait --for=condition=complete --timeout=10m job/bdaproject-stage2-job"
            }
        }
    }
}