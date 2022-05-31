import sys

import data_creator
import data_feeder
import pearsonbot

settings = data_feeder.get_settings()

def logic():
    print(f"""

        Running Backtest...
        Look out for logs

        settings:
            stop_loss:          {settings['sl']}
            take_profit:        {settings['tp']}
            std dev line (x):   {settings['x']}

            timeframe (mins):   {settings['timeframe']}
            min_linreg:         {settings['min_linreg']}

            trading start time: {settings['start_time']}
            trading end time:   {settings['end_time']}

""")
    pb = pearsonbot.PearsonBot(settings)
    pb.main()
    print(f"""

        Backtest Done!
        Check outputs folder for results

        hierarchy:
            all_trades      ->      all the trades executed by the bot sequentially
            entry           ->      only the ENTRY orders given by the bot sequentially
            exit            ->      only the EXIT orders given by the bot sequentially
            master          ->      contains the raw data along with linear regression values and std dev channel values
            pnl             ->      all the entries and exits side by side with pnl calculated
""")


def main():
    opt = str(input("Do you want to generate new data? [y/n]: "))
    if opt == 'y':
        print("Creating Data...\n")
        data_creator.main(settings['data_list'])
        logic()
    elif opt == 'n':
        logic()
    else:
        print("Please enter correct option")


if __name__ == "__main__":
    main()