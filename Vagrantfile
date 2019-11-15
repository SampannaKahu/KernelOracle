# IMPORTANT NOTES:
# For best results use a VirualBox version that matches the guest
# lsmod | grep -io vboxguest | xargs modinfo | grep -iw version

# provision script
#-----------------------------------------------------------------------
$bootstrap = <<BOOTSTRAP
export DEBIAN_FRONTEND=noninteractive
apt-get update

# label as reference environment
echo "export LKP_PROJECT_ENV=Y" >> /etc/environment

# install a lightweight graphical environment
apt-get -y install virtualbox-guest-dkms 
apt-get -y install virtualbox-guest-utils 
apt-get -y install virtualbox-guest-x11
apt-get -y install xorg sddm openbox
echo "[Autologin]" >> /etc/sddm.conf
echo "User=vagrant" >> /etc/sddm.conf
echo "Session=openbox.desktop" >> /etc/sddm.conf

# install a simple browser
apt-get -y install hv3

BOOTSTRAP
#-----------------------------------------------------------------------

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"
  config.vm.network "forwarded_port", guest: 80, host: 8080
  config.ssh.forward_agent = true
  config.ssh.forward_x11 = true

  # get rid of annoying console.log file and start the GUI 
  config.vm.provider "virtualbox" do |vb|
    vb.customize [ "modifyvm", :id, "--uartmode1", "disconnected" ]
    vb.gui = true
    vb.cpus = 1
  end

  # setup the VM
  config.vm.provision "shell", inline: $bootstrap
  
end

