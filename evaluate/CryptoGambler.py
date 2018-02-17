from misc.Logger import Logger


class CryptoGambler:
    def __init__(self,  current_price, crypto_wallet=1):
        self._start_wallet=crypto_wallet
        self._start_money = current_price
        self._crypto_wallet = crypto_wallet
        self._money = 0.0

    def evaluate_new_sample(self, real, predicted_diff):
        if predicted_diff < 0.0:
            self.sell_crypto(real)
        elif predicted_diff > 0.0:
            self.buy_crypto(real)

    def sell_crypto(self, real):
        if self._crypto_wallet > 0.0:
            self._money = self.convert_crypto_to_money(self._crypto_wallet, real)
            self._crypto_wallet = 0.0

    def buy_crypto(self, real):
        if self._money > 0.0:
            self._crypto_wallet = self.convert_money_to_crypto(self._money, real)
            self._money = 0.0

    def convert_crypto_to_money(self, crypto, price):
        return crypto * price

    def convert_money_to_crypto(self, money, price):
        return money / price

    def get_evaluation_results(self, real):
        if self._crypto_wallet == 0.0:
            self._crypto_wallet = self.convert_money_to_crypto(self._money, real)
        else:
            self._money = self.convert_crypto_to_money(self._crypto_wallet, real)

        Logger.header("Wallet : " + str(self._crypto_wallet) + " coins (" + str(self._money) + " €)")
        crypto_earnings = self._crypto_wallet - self._start_wallet
        money_earnings = self._money - self._start_money
        if crypto_earnings > 0.0:
            Logger.okgreen("Earnings : "+str(crypto_earnings)+" coins ("+str(money_earnings)+" €)")
        else:
            Logger.fail("Earnings : "+str(crypto_earnings)+" coins ("+str(money_earnings)+" €)")


