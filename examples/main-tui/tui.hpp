#pragma once

#include <ncurses.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstdlib>
#include <cstring>
#include <future>
#include <chrono>

#define WIDTH 30
#define HEIGHT 10 

int startx = 0;
int starty = 0;

char *choices[] = { 
                        "Choice 1",
                        "Choice 2",
                        "Choice 3",
                        "Choice 4",
                        "Exit",
                  };
int n_choices = sizeof(choices) / sizeof(char *);

inline void print_menu(WINDOW *menu_win, int highlight);

inline void print_stderr_win(WINDOW *win, int streamno);

struct tui_queue{
    bool dummy;
};

struct tui_channel{
    tui_queue user_input_queue;
    tui_queue llama_output_queue;
    tui_queue to_tui_signal_queue;
    tui_queue from_tui_signal_queue;

    std::condition_variable cv;
    std::mutex cm;
    std::atomic<bool> cc{false};

    int mem_stdout;
    int mem_stderr;
    int raw_stdout;
    int raw_stderr;
};

inline void tui_tui( std::shared_ptr<tui_channel> tui_comm_channel, std::atomic<bool> & run_flag );

std::shared_ptr<tui_channel> tui_thread_controller(bool run = true);

inline std::shared_ptr<tui_channel> tui_start();

inline std::shared_ptr<tui_channel> tui_stop();

void tui_cleanup();

#ifndef TUI_IMPL_HERE
#define TUI_IMPL_HERE

inline void print_menu(WINDOW *menu_win, int highlight)
{
    int x, y, i;    

    x = 2;
    y = 2;
    box(menu_win, 0, 0);
    for(i = 0; i < n_choices; ++i)
    {       if(highlight == i + 1) /* High light the present choice */
            {       wattron(menu_win, A_REVERSE); 
                    mvwprintw(menu_win, y, x, "%s", choices[i]);
                    wattroff(menu_win, A_REVERSE);
            }
            else
                    mvwprintw(menu_win, y, x, "%s", choices[i]);
            ++y;
    }
    wrefresh(menu_win);
}

inline void print_stderr_win(WINDOW *win, int streamno)
{
    static size_t readptr = 0;
    //box(win, 0, 0);
    scrollok(win, true);
    //wclear(win);
    static char buf[8192];
    memset(buf, 0, 8192);
    
    auto end = lseek(streamno, 0, SEEK_END);
    lseek(streamno, readptr, 0);

    auto cc = read(streamno, buf, 8191);
    readptr += cc;

    wprintw(win, "%s", buf );
    wrefresh(win);
}

inline void tui_tui( std::shared_ptr<tui_channel> tui_comm_channel, std::atomic<bool> & run_flag )
{
    auto channel = tui_comm_channel;

    auto nt = newterm(NULL, fdopen( channel->raw_stdout, "w" ), stdin);

    auto & run = run_flag;

    WINDOW *menu_win, *stderr_win_outer, *stderr_win;
    int highlight = 1;
    int choice = 0;
    int c;

    //initscr();
    clear();
    noecho();
    cbreak();       /* Line buffering disabled. pass on everything */

    startx = 0;//(80 - WIDTH) / 2;
    starty = 0;//(24 - HEIGHT) / 2;
            
    menu_win = newwin(LINES/2, COLS, starty, startx);
    stderr_win_outer = newwin(LINES/2, COLS, starty+(LINES/2), startx);
    box(stderr_win_outer, 0, 0);
    stderr_win = newwin( (LINES/2)-2, (COLS)-2, (starty+(LINES/2))+1, (startx)+1 );
    keypad(menu_win, TRUE);
    mvprintw(0, 0, "Use arrow keys to go up and down, Press enter to select a choice");
    refresh();
    print_menu(menu_win, highlight);

    uint64_t dummy=0;

    while(run || dummy <= 200)
    {
        wrefresh(stderr_win_outer);
        print_stderr_win(stderr_win, channel->mem_stderr);
        wtimeout(menu_win, 10);
        c = wgetch(menu_win);
        switch(c)
        {
        case ERR:
            //ncurses timeout?
            dummy++;
            break;
        case KEY_UP:
            if(highlight == 1)
                    highlight = n_choices;
            else
                    --highlight;
            break;
        case KEY_DOWN:
            if(highlight == n_choices)
                    highlight = 1;
            else 
                    ++highlight;
            break;
        case 10:
            choice = highlight;
            break;
        default:
            mvprintw(24, 0, "Charcter pressed is = %3d Hopefully it can be printed as '%c'", c, c);
            refresh();
            break;
        }
        print_menu(menu_win, highlight);
        if(choice != 0) /* User did a choice come out of the infinite loop */
            break;
    }       
    mvprintw(23, 0, "You chose choice %d with choice string %s\n", choice, choices[choice - 1]);
    clrtoeol();
    refresh();
    endwin();

    fprintf(stderr, "tuiend\n");
    channel->cv.notify_all();
    exit(0);
}

std::shared_ptr<tui_channel> tui_thread_controller(bool run)
{
    static std::atomic<bool> run_flag{true};
    static std::shared_ptr<tui_channel> tui_comm_channel(new tui_channel);
    static std::thread tui_thread;

    if( run && !tui_thread.joinable() )
    {
        //start
        //static char* mem_stdout_buf{nullptr};
        //static size_t mem_stdout_buf_size{0};
        static auto mem_stdout = memfd_create("mem_stdout", 0);//open_memstream(&mem_stdout_buf, &mem_stdout_buf_size);
        ftruncate(mem_stdout, 0);

        //static char* mem_stderr_buf{nullptr};
        //static size_t mem_stderr_buf_size{0};
        static auto mem_stderr = memfd_create("mem_stderr", 0);//open_memstream(&mem_stderr_buf, &mem_stderr_buf_size);
        ftruncate(mem_stderr, 0);

        static int raw_stdout = dup(fileno(stdout));
        static int raw_stderr = dup(fileno(stderr));

        if (-1 == dup2(mem_stdout, fileno(stdout)))
        {
            perror("cannot redirect stdout");
            //exit(255);
        }
        if (-1 == dup2(mem_stderr, fileno(stderr)))
        {
            perror("cannot redirect stderr");
            //exit(255);
        }

        tui_comm_channel->mem_stdout = dup(mem_stdout);
        tui_comm_channel->mem_stderr = dup(mem_stderr);
        tui_comm_channel->raw_stdout = raw_stdout;
        tui_comm_channel->raw_stderr = raw_stderr;

        lseek(tui_comm_channel->mem_stdout, 0, 0);
        lseek(tui_comm_channel->mem_stderr, 0, 0);

        // auto aa = fdopen( mem_stderr, "w" );
        // auto bb = fdopen( raw_stderr, "w" );

        // auto dd = fdopen( mem_stderr, "r" );

        // fprintf( stderr, "Test 1\n" );
        // fprintf( aa, "Test 2\n" );
        // fprintf( bb, "Test 3\n" );
        // fprintf( stdout, "Test 4\n" );

        // fflush(aa);

        // auto wr = fprintf(stderr, "Test");

        // char buf[16];
        // memset(buf, 0, 16);
        // lseek(mem_stderr, 0, 0);
        // auto cc = read(mem_stderr, buf, 15);
        //auto r = fprintf(bb, cc );

        run_flag = true;
        tui_thread = std::thread(tui_tui, tui_comm_channel, std::ref(run_flag));
        tui_thread.detach();
    }

    if( !run )
    {
        //stop
        run_flag = false;

        dup2(tui_comm_channel->raw_stderr, fileno(stderr));
        dup2(tui_comm_channel->raw_stdout, fileno(stdout));
    }

    return tui_comm_channel;
}

inline std::shared_ptr<tui_channel> tui_start(){
    std::atexit(tui_cleanup);
    return tui_thread_controller(true);
}

inline std::shared_ptr<tui_channel> tui_stop(){
    return tui_thread_controller(false);
}

void tui_cleanup()
{
    auto ch = tui_stop();
    std::unique_lock<std::mutex> lk(ch->cm);

    reset_shell_mode();

    fprintf(stderr, "Waiting for TUI to finish...\n");

    if(ch->cv.wait_for(lk, std::chrono::seconds(1), [&ch]{return ch->cc ? true: false;}))
    {
        fprintf(stderr, "TUI finished.\n");
    }
    else
    {
        fprintf(stderr, "TUI takes too long to finish, forcing exit.\n");
    }
}

#endif