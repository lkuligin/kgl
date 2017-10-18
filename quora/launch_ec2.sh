subnetId='subnet-394f2f61'
securityGroupID='sg-2bea964d'
keyName='as24-bi'
ami='ami-a8d2d7ce'
instanceType='m4.4xlarge'

export instanceId=$(aws ec2 run-instances --image-id $ami --count 1 --instance-type $instanceType --key-name $keyName --security-group-ids $securityGroupId --subnet-id $subnetId --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 128, \"VolumeType\": \"gp2\" } } ]" --query 'Instances[0].InstanceId' --output text)
aws ec2 create-tags --resources $instanceId --tags --tags Key=Name,Value=jupyterTest

aws ec2 wait instance-running --instance-ids $instanceId
sleep 10
export instanceUrl=$(aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
echo $instanceUrl

ssh -i ~/.aws/as24-bi.pem ubuntu@$instanceUrl "mkdir ~/test"
scp  -i ~/.aws/as24-bi.pem -r ./data ubuntu@e$instanceUrl:~/test 
scp  -i ~/.aws/as24-bi.pem -r ./utils ubuntu@$instanceUrl:~/test 
scp  -i ~/.aws/as24-bi.pem -r ./tmp ubuntu@$instanceUrl:~/test 

scp  -i ~/.aws/as24-bi.pem setup_instance.sh ubuntu@$instanceUrl 
ssh -i ~/.aws/as24-bi.pem ubuntu@$instanceUrl "./setup_instance.sh"