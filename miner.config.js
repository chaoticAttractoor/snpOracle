module.exports = {
  apps: [
    {
      name: 'miner',
      script: 'python3',
      args: './neurons/miner.py --netuid 28 --logging.debug --logging.trace --subtensor.network finney --wallet.name miner --wallet.hotkey default --axon.port 8999 --model chaotic_snp'
    },
  ],
};
