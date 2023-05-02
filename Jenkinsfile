pipeline {
    agent any

    environment {
        PROJECT_ID = "your_project_id"
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
                sh "kubectl wait --for=condition=complete --timeout=30m job/bdaproject-stage1-job"
            }
        }
        stage("Preprocessing and build model") {
            steps {
                sh "kubectl delete job bdaproject-stage2-job || true"
                sh "kubectl apply -f ./StackOverFlow-tag-predict-system/deployment/stage2-job.yaml"
                sh "kubectl wait --for=condition=complete --timeout=30m job/bdaproject-stage2-job"
            }
        }
        stage("Deploy model") {
            steps {
                sh "kubectl delete -f ./StackOverFlow-tag-predict-system/deployment/stage3-deployment.yaml || true"
                sh "kubectl apply -f ./StackOverFlow-tag-predict-system/deployment/stage3-deployment.yaml"
            }
        }
    }
}